from __future__ import annotations

import importlib.util
import json
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from threading import Lock
from time import time
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
DATASETS = UPLOADS / "datasets"
MODELS = UPLOADS / "models"
RUNS = UPLOADS / "runs"
REQUIRED_FIELDS = ("instruction", "input", "output", "workflow", "success_condition")
EXPECTED_RECORDS = 172
MAX_DATASET_BYTES = 50 * 1024 * 1024
MAX_MODEL_BYTES = 50 * 1024 * 1024 * 1024
MAX_MODEL_FILES = 4000
CHUNK_SIZE = 1024 * 1024

for directory in (DATASETS, MODELS, RUNS):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Automata Fine Tuning backend", docs_url=None, redoc_url=None)
JOBS: dict[str, dict] = {}
JOBS_LOCK = Lock()
TRAINING_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="automata-train")


class TrainingRequest(BaseModel):
    dataset_upload_id: str
    model_upload_id: str
    method: str = Field(default="lora", pattern=r"^(lora|qlora)$")
    epochs: int = Field(default=3, ge=1, le=20)
    learning_rate: float = Field(default=0.0002, gt=0, le=0.01)
    rank: int = Field(default=16, ge=1, le=64)
    alpha: int = Field(default=16, ge=1, le=128)
    dropout: float = Field(default=0, ge=0, le=0.5)
    batch_size: int = Field(default=1, ge=1, le=2)
    sequence_length: int = Field(default=512, ge=128, le=2048)


class TrainingTestRequest(BaseModel):
    command: str = Field(min_length=1, max_length=4000)


class HuggingFaceModelRequest(BaseModel):
    repo_id: str = Field(min_length=3, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
    revision: str = Field(default="main", min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")


def _safe_name(filename: str) -> str:
    name = Path(filename.replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name[:160] or "upload.jsonl"


def _hf_request(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Automata-Lab/1.0"})
    token = __import__("os").environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code in {401, 403}:
            raise HTTPException(status_code=403, detail="Hugging Face model is private or gated. Set HF_TOKEN and restart the backend.") from error
        if error.code == 404:
            raise HTTPException(status_code=404, detail="Hugging Face model repository or revision was not found.") from error
        raise HTTPException(status_code=502, detail=f"Hugging Face returned HTTP {error.code}.") from error
    except urllib.error.URLError as error:
        raise HTTPException(status_code=502, detail=f"Could not reach Hugging Face: {error.reason}") from error


def _hf_download_to_path(url: str, target: Path, remaining_bytes: int) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": "Automata-Lab/1.0"})
    token = __import__("os").environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as output:
            while chunk := response.read(CHUNK_SIZE):
                size += len(chunk)
                if size > remaining_bytes:
                    raise HTTPException(status_code=413, detail="Hugging Face model exceeds the local model size limit.")
                output.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except urllib.error.HTTPError as error:
        target.unlink(missing_ok=True)
        if error.code in {401, 403}:
            raise HTTPException(status_code=403, detail="Hugging Face model is private or gated. Set HF_TOKEN and restart the backend.") from error
        if error.code == 404:
            raise HTTPException(status_code=404, detail="Hugging Face model file or revision was not found.") from error
        raise HTTPException(status_code=502, detail=f"Hugging Face returned HTTP {error.code}.") from error
    except urllib.error.URLError as error:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail=f"Could not download from Hugging Face: {error.reason}") from error
    return size


def _download_huggingface_model(request: HuggingFaceModelRequest) -> dict:
    repo_id = request.repo_id
    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    tree_url = f"https://huggingface.co/api/models/{encoded_repo}/tree/{urllib.parse.quote(request.revision, safe='')}?recursive=true&expand=false"
    try:
        tree = json.loads(_hf_request(tree_url))
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=502, detail="Hugging Face returned an invalid repository listing.") from error
    if not isinstance(tree, list):
        raise HTTPException(status_code=502, detail="Hugging Face repository listing has an unexpected format.")

    files = [entry.get("path") for entry in tree if isinstance(entry, dict) and entry.get("type") == "file" and isinstance(entry.get("path"), str)]
    files = [path for path in files if not path.startswith(".git/")]
    if not files:
        raise HTTPException(status_code=422, detail="Hugging Face repository contains no downloadable files.")
    if len(files) > MAX_MODEL_FILES:
        raise HTTPException(status_code=413, detail=f"Hugging Face model contains more than {MAX_MODEL_FILES} files.")

    upload_id = uuid4().hex
    destination = MODELS / upload_id
    destination.mkdir(parents=True, exist_ok=False)
    total_size = 0
    try:
        for relative in files:
            normalized = PurePosixPath(relative)
            if normalized.is_absolute() or any(part in {"..", "."} for part in normalized.parts):
                raise HTTPException(status_code=502, detail="Hugging Face returned an unsafe file path.")
            url = f"https://huggingface.co/{encoded_repo}/resolve/{urllib.parse.quote(request.revision, safe='')}/{urllib.parse.quote(normalized.as_posix(), safe='/')}?download=true"
            target = destination / Path(*normalized.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            total_size += _hf_download_to_path(url, target, MAX_MODEL_BYTES - total_size)

        model_info = _validate_model_directory(destination)
        manifest = {
            "upload_id": upload_id,
            "source": "huggingface",
            "repo_id": repo_id,
            "revision": request.revision,
            "file_count": len(files),
            "size_bytes": total_size,
            "model_path": str(destination.relative_to(ROOT)),
            "files": files,
            **model_info,
        }
        (destination / "automata-upload.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _validate_dataset(path: Path) -> dict:
    count = 0
    missing_fields: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig") as dataset:
            for line_number, line in enumerate(dataset, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Invalid JSON on line {line_number}: {error.msg}",
                    ) from error
                if not isinstance(record, dict):
                    raise HTTPException(
                        status_code=422,
                        detail=f"Line {line_number} must contain a JSON object.",
                    )
                count += 1
                missing_fields.update(field for field in REQUIRED_FIELDS if field not in record)
    except UnicodeDecodeError as error:
        raise HTTPException(status_code=422, detail="Dataset must be UTF-8 encoded.") from error

    fields_valid = count > 0 and not missing_fields
    return {
        "record_count": count,
        "expected_records": EXPECTED_RECORDS,
        "required_fields": list(REQUIRED_FIELDS),
        "missing_fields": sorted(missing_fields),
        "schema_valid": fields_valid,
        "valid": fields_valid and count == EXPECTED_RECORDS,
    }


def _training_runtime() -> dict:
    cuda_available = False
    gpu_name = None
    gpu_memory_bytes = 0
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        pass
    return {
        "training_ready": cuda_available,
        "missing_packages": [],
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_memory_bytes": gpu_memory_bytes,
    }


def _persist_job(job: dict) -> None:
    job_dir = RUNS / job["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    temporary = job_dir / "job.json.tmp"
    temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
    temporary.replace(job_dir / "job.json")


def _restore_job(job_id: str) -> dict | None:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        return None
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is not None:
        return job

    job_file = RUNS / job_id / "job.json"
    if job_file.is_file():
        try:
            job = json.loads(job_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(job, dict) or job.get("job_id") != job_id:
            return None
        if job.get("status") in {"queued", "loading", "running"}:
            job.update(
                status="failed",
                error="Backend restarted before this training job completed.",
                message="Training was interrupted by a backend restart",
                updated_at=time(),
            )
            _persist_job(job)
        with JOBS_LOCK:
            JOBS.setdefault(job_id, job)
            return JOBS[job_id]

    adapter_dir = RUNS / job_id / "adapter"
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_config.is_file() or not adapter_weights.is_file():
        return None
    try:
        config = json.loads(adapter_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    model_upload_id = Path(str(config.get("base_model_name_or_path", ""))).name
    if not re.fullmatch(r"[0-9a-f]{32}", model_upload_id):
        return None
    created_at = adapter_dir.stat().st_mtime
    job = {
        "job_id": job_id,
        "status": "completed",
        "progress": 100,
        "step": 0,
        "total_steps": 0,
        "loss": None,
        "message": "Training completed; adapter and tokenizer saved",
        "model_upload_id": model_upload_id,
        "created_at": created_at,
        "updated_at": created_at,
        "output_path": str(adapter_dir.relative_to(ROOT)),
    }
    _persist_job(job)
    with JOBS_LOCK:
        JOBS.setdefault(job_id, job)
        return JOBS[job_id]


def _update_job(job_id: str, **changes) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(changes)
            JOBS[job_id]["updated_at"] = time()
            _persist_job(JOBS[job_id])


def _find_dataset(upload_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        raise HTTPException(status_code=400, detail="Invalid dataset upload ID.")
    matches = list(DATASETS.glob(f"{upload_id}_*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Dataset upload not found.")
    return matches[0]


def _find_model_directory(upload_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        raise HTTPException(status_code=400, detail="Invalid model upload ID.")
    model_root = MODELS / upload_id
    if not (model_root / "automata-upload.json").is_file():
        raise HTTPException(status_code=404, detail="Model upload not found.")
    config_path = next(model_root.rglob("config.json"), None)
    if config_path is None:
        raise HTTPException(status_code=422, detail="Uploaded model has no config.json.")
    return config_path.parent


def _validate_model_directory(model_root: Path) -> dict:
    config_path = next(model_root.rglob("config.json"), None)
    if config_path is None:
        raise HTTPException(status_code=422, detail="Model folder must contain config.json.")
    model_dir = config_path.parent
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=422, detail="Model config.json is not valid UTF-8 JSON.") from error
    if not isinstance(config, dict) or not isinstance(config.get("model_type"), str):
        raise HTTPException(status_code=422, detail="Model config.json must declare a model_type.")

    architectures = config.get("architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    if not isinstance(architectures, list) or any(not isinstance(name, str) for name in architectures):
        raise HTTPException(status_code=422, detail="Model architectures must be a list of names.")
    if architectures and not any("ForCausalLM" in architecture or "LMHeadModel" in architecture for architecture in architectures):
        raise HTTPException(status_code=422, detail="This app currently fine-tunes causal language models only.")

    files = [path for path in model_dir.rglob("*") if path.is_file()]
    filenames = {path.name.lower() for path in files}
    weight_files = [path for path in files if path.name.lower().endswith((".safetensors", ".bin"))]
    if not weight_files:
        raise HTTPException(status_code=422, detail="Model folder must contain .safetensors or .bin weights.")

    tokenizer_files = sorted(
        name
        for name in filenames
        if name in {"tokenizer.json", "tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "vocab.txt"}
    )
    if not tokenizer_files and not {"vocab.json", "merges.txt"}.issubset(filenames):
        raise HTTPException(
            status_code=422,
            detail="Model folder must include tokenizer.json, a SentencePiece model, vocab.txt, or vocab.json with merges.txt.",
        )
    tokenizer_json = next((path for path in files if path.name.lower() == "tokenizer.json"), None)
    if tokenizer_json is not None:
        try:
            if not isinstance(json.loads(tokenizer_json.read_text(encoding="utf-8")), dict):
                raise ValueError("tokenizer.json must contain an object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=f"Invalid tokenizer.json: {error}") from error

    shard_indexes = [path for path in files if path.name.lower() in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}]
    for index_path in shard_indexes:
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError) as error:
            raise HTTPException(status_code=422, detail=f"Invalid weight index {index_path.name}.") from error
        if not isinstance(weight_map, dict) or not weight_map:
            raise HTTPException(status_code=422, detail=f"Weight index {index_path.name} has no weight_map.")
        if any(not isinstance(name, str) or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts for name in weight_map.values()):
            raise HTTPException(status_code=422, detail=f"Weight index {index_path.name} contains invalid shard paths.")
        missing_shards = sorted({name for name in weight_map.values() if not (index_path.parent / name).is_file()})
        if missing_shards:
            raise HTTPException(status_code=422, detail=f"Model is missing weight shards: {', '.join(missing_shards[:5])}.")

    for weight_path in weight_files:
        if weight_path.stat().st_size == 0:
            raise HTTPException(status_code=422, detail=f"Model weight file {weight_path.name} is empty.")
        if weight_path.suffix.lower() == ".safetensors":
            try:
                with weight_path.open("rb") as weight_file:
                    header_length_bytes = weight_file.read(8)
                    if len(header_length_bytes) != 8:
                        raise ValueError("header is missing")
                    header_length = int.from_bytes(header_length_bytes, "little")
                    if header_length <= 0 or header_length > min(weight_path.stat().st_size - 8, 100 * 1024 * 1024):
                        raise ValueError("header length is invalid")
                    header = json.loads(weight_file.read(header_length))
                if not isinstance(header, dict):
                    raise ValueError("header must be a JSON object")
                tensor_entries = {name: details for name, details in header.items() if name != "__metadata__"}
                data_size = weight_path.stat().st_size - 8 - header_length
                if not tensor_entries:
                    raise ValueError("no tensors were declared")
                for tensor_name, details in tensor_entries.items():
                    offsets = details.get("data_offsets") if isinstance(details, dict) else None
                    shape = details.get("shape") if isinstance(details, dict) else None
                    if (
                        not isinstance(tensor_name, str)
                        or not isinstance(offsets, list)
                        or len(offsets) != 2
                        or any(not isinstance(offset, int) for offset in offsets)
                        or offsets[0] < 0
                        or offsets[1] < offsets[0]
                        or offsets[1] > data_size
                        or not isinstance(shape, list)
                        or any(not isinstance(dimension, int) or dimension < 0 for dimension in shape)
                    ):
                        raise ValueError(f"tensor metadata is invalid for {tensor_name}")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                raise HTTPException(status_code=422, detail=f"Invalid SafeTensors weight file {weight_path.name}: {error}.") from error
        elif weight_path.suffix.lower() == ".bin":
            with weight_path.open("rb") as weight_file:
                signature = weight_file.read(4)
            if not (signature.startswith(b"PK\x03\x04") or signature.startswith(b"\x80")):
                raise HTTPException(status_code=422, detail=f"PyTorch weight file {weight_path.name} has an invalid archive signature.")

    return {
        "model_type": config["model_type"],
        "architectures": architectures,
        "weight_file_count": len(weight_files),
        "tokenizer_files": tokenizer_files,
        "sharded": bool(shard_indexes),
    }


def _format_user_message(instruction: str, input_text: str, workflow: str, success_condition: str) -> str:
    return (
        f"Instruction\n{instruction}\n\n"
        f"Input\n{input_text}\n\n"
        f"Required workflow\n{workflow}\n\n"
        f"Success condition\n{success_condition}"
    )


def _format_training_record(record: dict, tokenizer=None) -> str:
    user_message = _format_user_message(
        record["instruction"],
        record["input"],
        str(record["workflow"]),
        record["success_condition"],
    )
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": record["output"]},
            ],
            tokenize=False,
        )
    return (
        f"### Instruction\n{record['instruction']}\n\n"
        f"### Input\n{record['input']}\n\n"
        f"### Required workflow\n{record['workflow']}\n\n"
        f"### Success condition\n{record['success_condition']}\n\n"
        f"### Response\n{record['output']}"
    )


def _run_training(job_id: str, dataset_path: Path, model_dir: Path, config: TrainingRequest) -> None:
    try:
        _update_job(job_id, status="loading", message="Loading model and tokenizer", progress=0)
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainerCallback,
        )
        from trl import SFTConfig, SFTTrainer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to PyTorch. Install a CUDA-enabled PyTorch build.")

        qlora_enabled = config.method == "qlora"
        if qlora_enabled:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception as error:
                raise RuntimeError("QLoRA requires bitsandbytes and an NVIDIA GPU. Install the required CUDA stack first.") from error
        else:
            quantization_config = None

        with dataset_path.open("r", encoding="utf-8-sig") as dataset_file:
            records = [json.loads(line) for line in dataset_file if line.strip()]
        if not _validate_dataset(dataset_path)["valid"]:
            raise RuntimeError("Training requires 172 records with all five required fields.")

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model_kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        if qlora_enabled:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), **model_kwargs)
        model.config.use_cache = False
        if not qlora_enabled:
            model.to("cuda")

        available_modules = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        target_modules = [
            name
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "c_attn", "c_proj", "c_fc")
            if name in available_modules
        ]
        if not target_modules:
            raise RuntimeError("No supported attention/feed-forward projection modules were found in this model.")
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.rank,
                lora_alpha=config.alpha,
                lora_dropout=config.dropout,
                target_modules=target_modules,
                bias="none",
            ),
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        dataset = Dataset.from_list([{"text": _format_training_record(row, tokenizer)} for row in records])
        output_dir = RUNS / job_id
        adapter_dir = output_dir / "adapter"
        output_dir.mkdir(parents=True, exist_ok=True)

        class ProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                logs = logs or {}
                progress = min(99, int(state.global_step * 100 / max(state.max_steps, 1)))
                _update_job(
                    job_id,
                    progress=progress,
                    step=state.global_step,
                    total_steps=state.max_steps,
                    loss=logs.get("loss"),
                    message=f"Training step {state.global_step} of {state.max_steps}",
                )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            callbacks=[ProgressCallback()],
            args=SFTConfig(
                output_dir=str(output_dir / "checkpoints"),
                num_train_epochs=config.epochs,
                learning_rate=config.learning_rate,
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=max(1, 4 // config.batch_size),
                gradient_checkpointing=True,
                fp16=True,
                optim="adamw_torch",
                logging_steps=1,
                save_strategy="epoch",
                save_total_limit=2,
                report_to="none",
                remove_unused_columns=True,
                seed=3407,
                dataset_text_field="text",
                max_seq_length=config.sequence_length,
            ),
        )
        method_label = "QLoRA" if qlora_enabled else "LoRA"
        _update_job(job_id, status="running", message=f"Fine-tuning with {method_label} on the local CUDA GPU")
        result = trainer.train()
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        _update_job(
            job_id,
            status="completed",
            progress=100,
            loss=result.training_loss,
            output_path=str(adapter_dir.relative_to(ROOT)),
            message="Training completed; adapter and tokenizer saved",
        )
    except Exception as error:
        message = str(error)
        if "out of memory" in message.lower():
            message = "GPU ran out of memory. Retry with a shorter sequence length or a smaller base model."
        _update_job(job_id, status="failed", error=message, message="Training failed")
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


async def _copy_upload(upload: UploadFile, destination: Path, byte_limit: int) -> int:
    size = 0
    with destination.open("wb") as target:
        while chunk := await upload.read(CHUNK_SIZE):
            size += len(chunk)
            if size > byte_limit:
                raise HTTPException(status_code=413, detail="Upload exceeds the size limit.")
            target.write(chunk)
    return size


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "storage": "local",
        "max_dataset_bytes": MAX_DATASET_BYTES,
        "max_model_bytes": MAX_MODEL_BYTES,
        **_training_runtime(),
    }


@app.get("/api/uploads")
def list_uploads() -> dict:
    datasets = []
    for path in sorted(DATASETS.glob("*"), key=lambda item: item.stat().st_mtime, reverse=True):
        if not path.is_file() or path.name.endswith(".part"):
            continue
        upload_id, _, filename = path.name.partition("_")
        if re.fullmatch(r"[0-9a-f]{32}", upload_id):
            datasets.append({"upload_id": upload_id, "filename": filename, **_validate_dataset(path)})

    models = []
    for model_dir in sorted(MODELS.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        manifest_path = model_dir / "automata-upload.json"
        if model_dir.is_dir() and manifest_path.is_file():
            models.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    for run_dir in RUNS.iterdir():
        if run_dir.is_dir():
            _restore_job(run_dir.name)
    with JOBS_LOCK:
        latest_job = max(JOBS.values(), key=lambda item: item["created_at"], default=None)
        latest_job = dict(latest_job) if latest_job else None
        completed_jobs = sorted(
            (dict(job) for job in JOBS.values() if job["status"] == "completed"),
            key=lambda job: job["created_at"],
            reverse=True,
        )
    model_by_id = {model["upload_id"]: model for model in models}
    trained_models = [
        {
            "job_id": job["job_id"],
            "model_upload_id": job["model_upload_id"],
            "model_name": model_by_id.get(job["model_upload_id"], {}).get("repo_id")
            or model_by_id.get(job["model_upload_id"], {}).get("model_type")
            or "Trained model",
            "created_at": job["created_at"],
        }
        for job in completed_jobs
    ]
    return {"datasets": datasets, "models": models, "latest_job": latest_job, "trained_models": trained_models}


@app.post("/api/uploads/dataset")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    if Path(file.filename or "").suffix.lower() not in {".jsonl", ".json"}:
        raise HTTPException(status_code=415, detail="Upload a .jsonl or .json dataset.")

    upload_id = uuid4().hex
    destination = DATASETS / f"{upload_id}_{_safe_name(file.filename or '')}"
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        size = await _copy_upload(file, temporary, MAX_DATASET_BYTES)
        validation = _validate_dataset(temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    return {
        "upload_id": upload_id,
        "filename": _safe_name(file.filename or ""),
        "size_bytes": size,
        "stored": True,
        **validation,
    }


@app.post("/api/uploads/model")
async def upload_model(
    files: list[UploadFile] = File(...),
    paths: str = Form(...),
) -> dict:
    try:
        relative_paths = json.loads(paths)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="Model file paths must be valid JSON.") from error
    if not isinstance(relative_paths, list) or len(relative_paths) != len(files):
        raise HTTPException(status_code=400, detail="Each model file needs a relative folder path.")
    if not files or len(files) > 4000:
        raise HTTPException(status_code=400, detail="Select a model folder with between 1 and 4000 files.")

    upload_id = uuid4().hex
    destination = MODELS / upload_id
    destination.mkdir(parents=True, exist_ok=False)
    total_size = 0
    saved_paths: list[str] = []
    try:
        for upload, raw_path in zip(files, relative_paths, strict=True):
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise HTTPException(status_code=400, detail="A model file has an invalid path.")
            normalized = PurePosixPath(raw_path.replace("\\", "/"))
            if normalized.is_absolute() or any(part in {"..", "."} for part in normalized.parts):
                raise HTTPException(status_code=400, detail="Model paths cannot escape the upload folder.")
            if normalized.parts and ":" in normalized.parts[0]:
                raise HTTPException(status_code=400, detail="Absolute model paths are not allowed.")
            target = destination / Path(*normalized.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            total_size += await _copy_upload(upload, target, MAX_MODEL_BYTES - total_size)
            saved_paths.append(normalized.as_posix())

        model_info = _validate_model_directory(destination)
        manifest = {
            "upload_id": upload_id,
            "file_count": len(saved_paths),
            "size_bytes": total_size,
            "model_path": str(destination.relative_to(ROOT)),
            "files": saved_paths,
            **model_info,
        }
        (destination / "automata-upload.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        for upload in files:
            await upload.close()

    return {
        "upload_id": upload_id,
        "file_count": len(saved_paths),
        "size_bytes": total_size,
        "stored": True,
        "model_path": str(destination.relative_to(ROOT)),
        "files": saved_paths,
        **model_info,
    }


@app.post("/api/uploads/model/huggingface")
def upload_huggingface_model(request: HuggingFaceModelRequest) -> dict:
    return _download_huggingface_model(request)


@app.get("/api/uploads/models/{upload_id}")
def get_model_upload(upload_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        raise HTTPException(status_code=400, detail="Invalid model upload ID.")
    manifest_path = MODELS / upload_id / "automata-upload.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="Model upload not found.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@app.post("/api/training/start")
def start_training(request: TrainingRequest) -> dict:
    runtime = _training_runtime()
    if not runtime["training_ready"]:
        raise HTTPException(status_code=503, detail="Local CUDA training is not ready. Install the training dependencies, then restart the app.")

    dataset_path = _find_dataset(request.dataset_upload_id)
    if not _validate_dataset(dataset_path)["valid"]:
        raise HTTPException(status_code=422, detail="Select a valid 172-record dataset before training.")
    model_dir = _find_model_directory(request.model_upload_id)
    with JOBS_LOCK:
        if any(job["status"] in {"queued", "loading", "running"} for job in JOBS.values()):
            raise HTTPException(status_code=409, detail="A training job is already running.")
        job_id = uuid4().hex
        now = time()
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "step": 0,
            "total_steps": 0,
            "loss": None,
            "message": "Training job queued",
            "model_upload_id": request.model_upload_id,
            "created_at": now,
            "updated_at": now,
        }
        _persist_job(JOBS[job_id])
    TRAINING_POOL.submit(_run_training, job_id, dataset_path, model_dir, request)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/training/{job_id}")
def get_training_job(job_id: str) -> dict:
    job = _restore_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    return job


@app.get("/api/training/{job_id}/download")
def download_trained_model(job_id: str):
    job = _restore_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail="Training must be completed before downloading the adapter.")

    adapter_dir = ROOT / job["output_path"]
    if not adapter_dir.is_dir():
        raise HTTPException(status_code=404, detail="The completed adapter folder was not found.")

    output_dir = ROOT / "uploads" / "downloads"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{job_id}_adapter.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(adapter_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(adapter_dir.parent))

    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=f"{job_id}_adapter.zip",
    )


@app.delete("/api/training/{job_id}")
def delete_trained_model(job_id: str) -> dict:
    job = _restore_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")

    with JOBS_LOCK:
        JOBS.pop(job_id, None)

    run_dir = RUNS / job_id
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    return {"deleted": True, "job_id": job_id}


@app.post("/api/training/{job_id}/test")
def test_trained_model(job_id: str, request: TrainingTestRequest) -> dict:
    job = _restore_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail="Wait for training to complete before testing the model.")
    if not _training_runtime()["training_ready"]:
        raise HTTPException(status_code=503, detail="CUDA training dependencies are no longer available.")

    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    adapter_dir = ROOT / job["output_path"]
    model_dir = _find_model_directory(job["model_upload_id"])
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_dir),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    model.eval()
    instruction = "Follow the Automata workflow for this command."
    workflow = "List the ordered actions and checks."
    success_condition = "State how success will be verified."
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": _format_user_message(instruction, request.command.strip(), workflow, success_condition),
            }],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = (
            f"### Instruction\n{instruction}\n\n"
            f"### Input\n{request.command.strip()}\n\n"
            f"### Required workflow\n{workflow}\n\n"
            f"### Success condition\n{success_condition}\n\n"
            "### Response\n"
        )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    del model
    torch.cuda.empty_cache()
    return {"response": response}


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/use-model")
def use_model_page() -> FileResponse:
    return FileResponse(ROOT / "use_model.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

# Automata Lab

## Start the local website and upload API

From this folder, install the backend dependencies and start the service:

```powershell
python -m pip install -r requirements.txt
python backend.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The API listens on localhost only. Dataset uploads are checked for JSONL syntax, the five Automata fields, and exactly 172 records. Model uploads are checked for a parseable causal-LM `config.json`, tokenizer assets, weight files, and the completeness of any declared weight shards; unsupported architectures and incomplete folders are rejected before training.

After a training run completes, use [http://127.0.0.1:8000/use-model](http://127.0.0.1:8000/use-model) to open the separate chat page. Choose any completed trained run from the model selector; the newest completed run is selected by default. Completed adapters are recovered from local run files after a backend restart.

## Enable local GPU training

The Train step needs the CUDA training stack in addition to the upload API:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r training-requirements.txt
python backend.py
```

The app enables Start local training only when PyTorch detects CUDA. Training uses the uploaded Transformers model, the validated 172-row dataset, LoRA, and the local CUDA GPU. Completed adapters and checkpoints are written under `uploads/runs/`.

This computer has an RTX 2050 with 4 GB VRAM. Start with a 1B model, batch size 1, and 512-token sequences; larger models may run out of memory. Actual training could not be exercised without the training dependencies and user-provided model and dataset.

Uploaded files are stored under `uploads/` on this computer and are excluded from Git. Training and inference stay on this device.

## Download a model from Hugging Face

In the Model step, enter a public repository ID such as `Qwen/Qwen2.5-0.5B-Instruct` and click **Download and validate model**. The backend streams the repository files to local storage, validates the causal-LM config, tokenizer, weights, and shard indexes, then makes the model available to the Train step. Private or gated repositories require an `HF_TOKEN` environment variable before starting the backend.

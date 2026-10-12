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

## Deploy the website to Render

Render's standard web-service instances do not provide a CUDA GPU. The website can still let each user train on their own Windows GPU by connecting the browser to the complete local backend. Dataset uploads, model files, training jobs, and trained adapters then stay on that user's computer; Render cannot access those local files or run the training job.

On the GPU computer, install the local dependencies and CUDA-enabled PyTorch as described above. Set `AUTOMATA_WEB_ORIGINS` to the exact HTTPS origin of the deployed website (no path or trailing slash), then start the local backend:

   ```powershell
   $env:AUTOMATA_WEB_ORIGINS = "https://your-service.onrender.com"
   .\.venv\Scripts\python.exe backend.py
   ```

Open the Render site, choose **Local runtime**, and click **Connect PC to train** using `http://127.0.0.1:8000`. Confirm the website's connection request and allow the browser's local-network permission if prompted. The saved local-backend address is not reconnected automatically; you must approve the connection each time you open the website. Once connected, upload the dataset and base model in the website; the files are sent to the local backend. Keep the backend running for uploads, training, testing, and downloads.

The backend binds only to `127.0.0.1`. Cross-origin access is disabled unless `AUTOMATA_WEB_ORIGINS` is set, and then only the exact configured HTTPS website origin is allowed. Disconnecting stops the website from using the local backend. Do not bind the backend to `0.0.0.0` or expose it to the internet. Users who open the website on devices without the local backend still get the website, but cannot train until they run the backend on a CUDA-enabled NVIDIA PC.

## Download a model from Hugging Face

In the Model step, enter a public repository ID such as `Qwen/Qwen2.5-0.5B-Instruct` and click **Download and validate model**. The backend streams the repository files to local storage, validates the causal-LM config, tokenizer, weights, and shard indexes, then makes the model available to the Train step. Private or gated repositories require an `HF_TOKEN` environment variable before starting the backend.

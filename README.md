# Local Watermark Remover

## Environment setup

Initialize Python env from lock-style dependency list:

```bash
cd <repo-dir>
./scripts/bootstrap_env.sh
```

Optional: initialize env and pre-download required models:

```bash
DOWNLOAD_MODELS=1 ./scripts/bootstrap_env.sh
```

Optional: include extra optional models in manifest:

```bash
DOWNLOAD_MODELS=1 INCLUDE_OPTIONAL_MODELS=1 ./scripts/bootstrap_env.sh
```

Files:

- `requirements.txt`: Python package install list
- `models/manifest.json`: model install list
- `scripts/download_models.py`: model downloader
- `scripts/bootstrap_env.sh`: one-shot environment bootstrap

## Run

```bash
cd <repo-dir>
./run.sh web 8788
```

Open:

- `http://127.0.0.1:8788`
- API docs: `http://127.0.0.1:8788/docs`

## CLI usage

Direct CLI removal:

```bash
./run.sh cli --input 2.png --output 2_out.png --mask-output 2_mask.png
```

With manual mask:

```bash
./run.sh cli --input 2.png --output 2_out.png --manual-mask mask2.png
```

## API usage

1. Upload image:

```bash
curl -F "file=@2.png" http://127.0.0.1:8788/api/jobs
```

2. Optional auto mask:

```bash
curl -X POST http://127.0.0.1:8788/api/jobs/<job_id>/auto-mask -H "Content-Type: application/json" -d "{}"
```

3. Start processing:

```bash
curl -X POST http://127.0.0.1:8788/api/jobs/<job_id>/process \
  -H "Content-Type: application/json" \
  -d '{"mask_source":"auto","steps":20,"guidance_scale":7.0,"seed":42}'
```

4. Query status:

```bash
curl http://127.0.0.1:8788/api/jobs/<job_id>
```

5. Download output:

```bash
curl -o output.png http://127.0.0.1:8788/api/jobs/<job_id>/assets/output
```

## Project structure

- `webapp/`: Web backend + frontend
- `watermark_remover.py`: standalone CLI script
- `data/jobs/`: runtime job data (uploaded images, masks, outputs, job.json)
- `scripts/cleanup_jobs.sh`: clean runtime job data

## `data/jobs` cleanup

`data/jobs` is runtime data, not source code.

- Clean all jobs:
```bash
./scripts/cleanup_jobs.sh all
```

- Clean jobs older than N days:
```bash
./scripts/cleanup_jobs.sh 7
```

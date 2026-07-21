# Build Face Engine With Nuitka

This documents the successful Docker-based Nuitka build for `face-engine`.

## Environment

- Project root: `/project`
- Docker service/container used: `jetson-face-engine`
- Entrypoint built: `run.py`
- Output artifact: `build/face-engine.bin`

## Start the Docker Runtime

Run from the repo root:

```bash
docker compose up -d
docker compose ps
```

Make sure `jetson-face-engine` is running before building.

## Install Build Dependencies In Docker

```bash
docker exec jetson-face-engine bash -c '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq patchelf ccache gcc g++
  python3 -m pip install --no-cache-dir nuitka ordered-set zstandard
'
```

## Build Command

Run inside the container from `/project`.

**Working mode (accelerated, non-standalone):** do **not** use `--onefile`.
`--onefile` either hangs while analyzing `torch` on Jetson, or isolates Python so system
packages (`uvicorn`, `torch`, …) are not importable. Accelerated mode matches the caveat
below: binary still needs the same Docker runtime.

```bash
docker exec jetson-face-engine bash -c 'cd /project && mkdir -p build weights && python3 -m nuitka \
  --output-dir=build \
  --output-filename=face-engine.bin \
  --include-data-dir=resources=resources \
  --include-data-dir=weights=weights \
  --module-parameter=torch-disable-jit=yes \
  --enable-plugin=no-qt \
  --follow-import-to=app \
  run.py'
```

## Expected Result

Successful build artifact:

```bash
/project/build/face-engine.bin
```

From the host workspace:

```bash
/home/atin/ntiendung/face-engine/build/face-engine.bin
```

Size ~6.6MB (accelerated). Exit code `0`, message: `Successfully created 'build/face-engine.bin'`.

## Verification

```bash
docker exec jetson-face-engine ls -lh /project/build/face-engine.bin
docker exec jetson-face-engine ldd /project/build/face-engine.bin

# Smoke (stop API first so port 17103 is free)
docker compose stop
docker compose run --rm --no-deps --entrypoint bash jetson-face-engine \
  -c 'cd /project && timeout 50 ./build/face-engine.bin'
docker compose up -d
```

What was verified:

- Nuitka finished with exit code `0`
- `face-engine.bin` created in `build/`
- Binary started Uvicorn, loaded InsightFace (CUDA), synced persons, `Application startup complete`

## Important Caveat

This build is non-standalone in practice.

That means:

- it is intended to run in the same Docker/runtime environment
- Python/native dependencies are still expected from that environment
- it is suitable for copying to another machine only if that machine provides a compatible runtime image/environment

## Recommended Copy Targets

Copy these together if you want to reuse the result on another machine:

- `build/face-engine.bin`
- `resources/`
- `weights/`
- the matching Docker image/runtime environment

## Notes

- Prefer accelerated mode (`--follow-import-to=app`, no `--onefile`) on Jetson
- `resources` and `weights` must be included (`weights/` may be empty)
- Full `--onefile` that follows `torch` is not practical on this device

"""Deploy the REAL (GPU) service on Modal.

    modal deploy sapient_serving/modal_app.py

This is the only place that runs the real model. It:
  * builds a GPU image with the [real] deps + the private sapient1 package,
  * mounts secrets (HF token to pull weights; the hashed key store) from Modal's
    secret store — never from code,
  * serves the same FastAPI app with SAPIENT_ENGINE=real,
  * scales to zero when idle so idle GPUs cost nothing.

NOTE: Modal's Python API names shift between versions — treat the calls below as a
template and align them with your installed `modal` version. This module is never
imported by local mock dev.
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi", "uvicorn", "pydantic>=2", "python-multipart", "numpy", "matplotlib",
        "torch", "transformers", "scipy", "nilearn", "huggingface_hub", "safetensors",
    )
    # Install the private model package on the build box (needs a git token secret):
    # .run_commands(
    #     "pip install 'sapient1 @ git+https://github.com/The-Sapient-Company/"
    #     "sapient-models.git#subdirectory=sapient1'"
    # )
    .add_local_python_source("sapient_serving")
)

app = modal.App("sapient-1-encoding-api")


@app.function(
    image=image,
    gpu="A100-40GB",
    secrets=[
        modal.Secret.from_name("sapient-hf"),    # provides HF_TOKEN
        modal.Secret.from_name("sapient-keys"),  # provides the hashed key store / its path
    ],
    timeout=600,
    scaledown_window=300,  # idle -> scale to zero
    min_containers=0,
)
@modal.asgi_app()
def fastapi_app():
    import os

    os.environ["SAPIENT_ENGINE"] = "real"
    from sapient_serving.api.app import create_app

    return create_app()

import subprocess  # nosec B404
import os
import shutil
import pytest
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_wwb(args):
    logger.info(" ".join(["wwb"] + args))
    result = subprocess.run(["wwb"] + args, capture_output=True, text=True)
    logger.info(result)
    return result


@pytest.mark.parametrize(
    ("model_id", "model_type", "backend"),
    [
        ("hf-internal-testing/tiny-stable-diffusion-torch", "sd", "hf"),
        ("hf-internal-testing/tiny-stable-diffusion-torch", "sd", "openvino"),
        ("hf-internal-testing/tiny-stable-diffusion-xl-pipe", "sd-xl", "hf"),
    ],
)
def test_image_model_types(model_id, model_type, backend):
    GT_FILE = "test_sd.json"
    wwb_args = [
        "--base-model",
        model_id,
        "--target-model",
        model_id,
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
    ]
    if backend == "hf":
        wwb_args.append("--hf")

    result = run_wwb(wwb_args)
    print(f"WWB result: {result}, {result.stderr}")

    try:
        os.remove(GT_FILE)
    except OSError:
        pass
    shutil.rmtree("reference", ignore_errors=True)
    shutil.rmtree("target", ignore_errors=True)

    assert result.returncode == 0
    assert "Metrics for model" in result.stderr
    assert "## Reference text" not in result.stderr


@pytest.mark.parametrize(
    ("model_id", "model_type", "backend"),
    [
        ("hf-internal-testing/tiny-stable-diffusion-torch", "sd", "hf"),
    ],
)
def test_image_custom_dataset(model_id, model_type, backend):
    GT_FILE = "test_sd.json"
    wwb_args = [
        "--base-model",
        model_id,
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--dataset",
        "google-research-datasets/conceptual_captions",
        "--dataset-field",
        "caption",
    ]
    if backend == "hf":
        wwb_args.append("--hf")

    result = run_wwb(wwb_args)

    assert os.path.exists(GT_FILE)

    try:
        os.remove(GT_FILE)
    except OSError:
        pass
    shutil.rmtree("reference", ignore_errors=True)

    assert result.returncode == 0

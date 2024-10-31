# LLMs benchmarking sample

This sample script demonstrates how to benchmark an LLMs in OpenVINO GenAI. The script includes functionality for warm-up iterations, generating text, and calculating various performance metrics.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

It's not required to install [../../export-requirements.txt](../../export requirements.txt) for deployment if the model has already been exported.

```sh
pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --trust-remote-code --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```

## Usage

Follow [Get Started with Samples](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/get-started-demos.html) to run the sample.

```sh
benchmark_genai [OPTIONS]
```

### Options

- `-m, --model`: Path to the model and tokenizers base directory.
- `-p, --prompt` (default: `"The Sky is blue because"`): The prompt to generate text.
- `-nw, --num_warmup` (default: `1`): Number of warmup iterations.
- `-mt, --max_new_tokens` (default: `20`): Number of warmup iterations.
- `-n, --num_iter` (default: `3`): Number of iterations.
- `-d, --device` (default: `"CPU"`): Device to run the model on.

### Output:

```
benchmark_genai -m TinyLlama-1.1B-Chat-v1.0 -n 10
```

```
Load time: 3405.69 ms
Generate time: 1430.77 ± 3.04 ms
Tokenization time: 0.51 ± 0.02 ms
Detokenization time: 0.37 ± 0.01 ms
TTFT: 81.60 ± 0.54 ms
TPOT: 71.52 ± 2.72 ms
Throughput tokens/s: 13.98 ± 0.53
```

For more information how performance metrics are calculated please follow [performance-metrics tutorial](../../../src/README.md#performance-metrics).

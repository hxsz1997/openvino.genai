# Benchmarking Script for Minicpm-v-2_6

This script supports three methods of inference using the Minicpm-v-2_6.

## 1. Port from [Notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/minicpm-v-multimodal-chatbot)
```bash
python benchmark.py -m <model> -d <device> -r <report_csv> -f <framework> -p <prompt text> -n <num_iters> --use_notebook
```
The model used here needs to be converted to IR format as mentioned in the [Notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/minicpm-v-multimodal-chatbot).

## 2. Support openvino_genai API
```bash
python benchmark.py -m <model> -d <device> -r <report_csv> -f <framework> -p <prompt text> -n <num_iters> --genai
```
The model used here uses optimum-cli tool.

## 3. Support optimum.intel API
```bash
python benchmark.py -m <model> -d <device> -r <report_csv> -f <framework> -p <prompt text> -n <num_iters>
```
The model used here uses optimum-cli tool.

### TODO
vision_latency and sampler_latency will be added later.
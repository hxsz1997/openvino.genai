# speculative_decoding_lm C++ sample that supports most popular models like LLaMA 3

Speculative decoding (or [assisted-generation](https://huggingface.co/blog/assisted-generation#understanding-text-generation-latency) in HF terminology) is a recent technique, that allows to speed up token generation when an additional smaller draft model is used alonside with the main model.

Speculative decoding works the following way. The draft model predicts the next K tokens one by one in an autoregressive manner, while the main model validates these predictions and corrects them if necessary. We go through each predicted token, and if a difference is detected between the draft and main model, we stop and keep the last token predicted by the main model. Then the draft model gets the latest main prediction and again tries to predict the next K tokens, repeating the cycle.

This approach reduces the need for multiple infer requests to the main model, enhancing performance. For instance, in more predictable parts of text generation, the draft model can, in best-case scenarios, generate the next K tokens that exactly match the target. In that case they are validated in a single inference request to the main model (which is bigger, more accurate but slower) instead of running K subsequent requests. More details can be found in the original paper https://arxiv.org/pdf/2211.17192.pdf, https://arxiv.org/pdf/2302.01318.pdf

This example showcases inference of text-generation Large Language Models (LLMs): `chatglm`, `LLaMA`, `Qwen` and other models with the same signature. The application doesn't have many configuration options to encourage the reader to explore and modify the source code.  Loading `openvino_tokenizers` to `ov::Core` enables tokenization. Run `optimum-cli` to generate IRs for the samples. There is also a Jupyter [notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-chatbot) which provides an example of LLM-powered Chatbot in Python.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

It's not required to install [../../export-requirements.txt](../../export requirements.txt) for deployment if the model has already been exported.

```sh
pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --trust-remote-code --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
optimum-cli export openvino --trust-remote-code --model meta-llama/Llama-2-7b-chat-hf Llama-2-7b-chat-hf
```

## Run

Follow [Get Started with Samples](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/get-started-demos.html) to run the sample.

`speculative_decoding_lm TinyLlama-1.1B-Chat-v1.0 Llama-2-7b-chat-hf "Why is the Sun yellow?"`


Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM. For example, the model meta-llama/Llama-2-13b-chat-hf can benefit from being run on a dGPU. Modify the source code to change the device for inference to the GPU.

See https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md#supported-models for the list of supported models.

### Troubleshooting

#### Unicode characters encoding error on Windows

Example error:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u25aa' in position 0: character maps to <undefined>
```

If you encounter the error described in the example when sample is printing output to the Windows console, it is likely due to the default Windows encoding not supporting certain Unicode characters. To resolve this:
1. Enable Unicode characters for Windows cmd - open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
2. Enable UTF-8 mode by setting environment variable `PYTHONIOENCODING="utf8"`.

from llm_bench_utils.ov_model_classes import OvModelForCausalLMWithEmb, OvMiniCPMV, init_model
import openvino as ov
from openvino.runtime import Core
from pathlib import Path

model_path = "/home/huang/hx_repo/openvino_notebooks/notebooks/minicpm-v-multimodal-chatbot/MiniCPM-V-2_6/"
core = ov.Core()
model_path = Path(model_path)
image_emb_path = Path("image_encoder.xml")
resampler_path = Path("resampler.xml")
llm_path = Path("language_model_int4")

ov_model, tokenizer = init_model(core, model_path, llm_path, image_emb_path, resampler_path, "GPU")


import requests
from PIL import Image

url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
image = Image.open(requests.get(url, stream=True).raw)
question1 = "What is unusual on this image?"
question2 = "Can you describe the scene in this image?"

tokenizer = ov_model.processor.tokenizer

# msgs = [{"role": "user", "content": question1}]
msgs = [
    [{"role": "user", "content": [image, question1]}],
    [{"role": "user", "content": [image, question1]}]
]

print("Answer:")
res,_,_,_,_,_ = ov_model.chat(image=None, msgs=msgs, context=None, tokenizer=tokenizer, sampling=False, stream=False, max_new_tokens=50)

generated_text = ""
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end="")
import torch
from transformers import AutoTokenizer
from model.llama_anns import LlamaForCausalLM

MODEL_PATH = "meta-llama/Llama-2-7b-hf"  # 替换为本地路径或 HuggingFace model id

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

prompt = "please explain LLM"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

import torch
from transformers import AutoTokenizer, AutoModelForCasualLM

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "pt-3.5-turbo"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCasualLM.from_pretrained(model_name).to(device)

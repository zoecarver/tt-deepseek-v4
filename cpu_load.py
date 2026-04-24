from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V4-Flash", dtype="auto")
print("MODEL_LOADED")
print("dtype:", next(model.parameters()).dtype)
print("num_params:", sum(p.numel() for p in model.parameters()))
print("config:", model.config)

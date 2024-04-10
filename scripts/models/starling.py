import torch
import transformers

from torch.cuda.amp import autocast

torch.cuda.empty_cache()

torch.cuda.memory_summary(device=None, abbreviated=False)

# Inside your training loop
with autocast():
    # Load tokenizer and model on the specified device
    tokenizer = transformers.AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
    model = transformers.AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", low_cpu_mem_usage=True, device_map="cuda")

def generate_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids,
        max_length=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response_ids = outputs[0]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text


## Multi-turn conversation
prompt = "Hello."
follow_up_question =  "Do you like sports? If so, which ones?"
response = ""
multi_turn_prompt = f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant: {response}<|end_of_turn|>GPT4 Correct User: {follow_up_question}<|end_of_turn|>GPT4 Correct Assistant:"
response_text = generate_response(multi_turn_prompt)
print("Multi-turn conversation response:", response_text)
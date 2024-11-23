import os
import pandas as pd
import kagglehub
import torch
from datasets import Dataset
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# Load the fine-tuned model for inference
tokenizer = LlamaTokenizer.from_pretrained('fine-tuned-llama')
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set

model = LlamaForCausalLM.from_pretrained(
    'fine-tuned-llama',
    device_map='auto',
    torch_dtype=torch.float16,
)
model.eval()

# Generate text with the fine-tuned model
input_text = (
    "Song: My Sweet Lord\n"
    "Artist: George Harrison\n"
    "Album: All Things Must Pass (Remastered)\n"
    "Popularity: 0\n"
    "Lyrics:\n"
)

input_ids = tokenizer.encode(input_text, return_tensors='pt')

attention_mask = torch.ones_like(input_ids)

input_ids = input_ids.to(model.device)
attention_mask = attention_mask.to(model.device)

# Generate text with the model
with torch.no_grad():
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=300,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
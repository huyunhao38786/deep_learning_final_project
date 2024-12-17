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

path = kagglehub.dataset_download("suraj520/music-dataset-song-information-and-lyrics")

print("Path to dataset files:", path)

# List dataset files
dataset_files = os.listdir(path)
print("Dataset files:", dataset_files)

# Find the CSV file in the downloaded dataset
csv_files = [file for file in dataset_files if file.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in the dataset directory.")
dataset_path = os.path.join(path, csv_files[0])

# Load the dataset
data = pd.read_csv(dataset_path)

# Explore the dataset
print("Dataset columns:", data.columns)
print(data.head())

# Standardize column names to lowercase (optional but recommended)
data.columns = data.columns.str.lower()
print("Standardized columns:", data.columns)

# Check if all required columns exist
required_columns = ['name', 'artist', 'album', 'popularity', 'lyrics']
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"The dataset does not contain a '{col}' column.")

# Remove rows with missing values in required columns
data = data.dropna(subset=required_columns)

# Optional: Clean the text in each column
def clean_text(text):
    return str(text).strip()

for col in required_columns:
    data[col] = data[col].apply(clean_text)

# Create a new 'text' column that combines all required columns
def create_text(row):
    return (
        f"Song: {row['name']}\n"
        f"Artist: {row['artist']}\n"
        f"Album: {row['album']}\n"
        f"Popularity: {row['popularity']}\n"
        f"Lyrics:\n{row['lyrics']}"
    )

data['text'] = data.apply(create_text, axis=1)

# Convert the 'text' column to a list
texts = data['text'].tolist()

# Create a Hugging Face Dataset
dataset = Dataset.from_dict({'text': texts})

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf')

# If tokenizer doesn't have a pad token, add it
tokenizer.pad_token = tokenizer.eos_token

# Define the tokenization function
def tokenize_function(example):
    return tokenizer(
        example['text'],
        truncation=True,
        max_length=512,
        padding='max_length',
        return_special_tokens_mask=False,
    )

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Load the model in 8-bit precision
model = LlamaForCausalLM.from_pretrained(
    'meta-llama/Llama-2-13b-hf',
    load_in_8bit=True,
    device_map='auto',
)

# Prepare the model for PEFT with LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


training_args = TrainingArguments(
    output_dir='./results-13b',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    save_total_limit=2,
    save_steps=500,
    evaluation_strategy='no',
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the model and tokenizer
model.save_pretrained('fine-tuned-llama-13b')
tokenizer.save_pretrained('fine-tuned-llama-13b')


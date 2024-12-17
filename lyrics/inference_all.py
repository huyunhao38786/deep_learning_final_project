import os
import pandas as pd
import kagglehub
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from Levenshtein import distance as levenshtein_distance
import difflib  # For computing LCS
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import random
import openai
from openai import OpenAI
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


client = OpenAI()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
# Download the latest version of the dataset
path = kagglehub.dataset_download("suraj520/music-dataset-song-information-and-lyrics")
print("Path to dataset files:", path)

# List dataset files
dataset_files = os.listdir(path)
print("Dataset files:", dataset_files)

DATASET_DIR = './dataset'
os.makedirs(DATASET_DIR, exist_ok=True)

# Find the CSV file in the downloaded dataset
csv_files = [file for file in dataset_files if file.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in the dataset directory.")
dataset_path = os.path.join(path, csv_files[0])

# Load the dataset
data = pd.read_csv(dataset_path)

# Standardize column names to lowercase
data.columns = data.columns.str.lower()

# Check if all required columns exist
required_columns = ['name', 'artist', 'album', 'popularity', 'lyrics']
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"The dataset does not contain a '{col}' column.")

# Remove rows with missing values in required columns
data = data.dropna(subset=required_columns)
# data = data.head(100)

# Clean the text in each column
def clean_text(text):
    return str(text).strip()

for col in required_columns:
    data[col] = data[col].apply(clean_text)

data = data.reset_index(drop=True)
data = data.iloc[380:]

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained('fine-tuned-llama')
tokenizer.pad_token = tokenizer.eos_token

# Load the fine-tuned model
model = LlamaForCausalLM.from_pretrained(
    'fine-tuned-llama',
    device_map='auto',
    torch_dtype=torch.float16,
    # load_in_8bit=True
    # offload_folder="offload_dir",
    # trust_remote_code=True,
    # low_cpu_mem_usage=True
)
model.eval()

# # Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def compute_lcs(a, b):
    """Computes the length of the Longest Common Subsequence between two strings."""
    matcher = difflib.SequenceMatcher(None, a, b)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    return match.size

def paraphrase_text(text):
    """Use the OpenAI API to paraphrase the given text."""
    if not text.strip():
        return text  # If text is empty, return as is
    
    prompt = f"Paraphrase the following text in a coherent way:\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that paraphrases text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200
    )
    paraphrased = response.choices[0].message.content.strip()
    return paraphrased

def get_random_line_segment(lyrics, portion=0.3, from_start=True):
    """Select a random non-empty line from the lyrics and return either the first or last portion% of it."""
    lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
    if not lines:
        return ""  # If no lines available, return empty string

    chosen_line = random.choice(lines)
    length = len(chosen_line)
    segment_length = max(1, int(length * portion))
    if from_start:
        return chosen_line[:segment_length]
    else:
        return chosen_line[-segment_length:]

# Checkpoint settings
CHECKPOINT_INTERVAL = 10
CHECKPOINT_FILE = os.path.join(DATASET_DIR, "checkpoint_results.csv")

# If checkpoint file exists, load it and continue from there
if os.path.exists(CHECKPOINT_FILE):
    checkpoint_data = pd.read_csv(CHECKPOINT_FILE, index_col=0)
    # Merge with original data to see which indices are completed
    # We assume the dataset hasn't changed order. If it has, you'd need a stable identifier.
    completed_indices = set(checkpoint_data.index)
    data = data.reset_index(drop=True)
    # Append checkpoint columns to data if not present
    for c in ['generated_lyrics_paraphrase', 'lcs_paraphrase', 'levenshtein_paraphrase',
              'generated_lyrics_reverse', 'lcs_reverse', 'levenshtein_reverse']:
        if c not in data.columns:
            data[c] = None
    # Update data rows from checkpoint_data
    for c in checkpoint_data.columns:
        if c in data.columns:
            data.loc[checkpoint_data.index, c] = checkpoint_data[c].values
else:
    completed_indices = set()


# Initialize lists only if not already done
if 'generated_lyrics_paraphrase' not in data.columns:
    data['generated_lyrics_paraphrase'] = None
if 'lcs_paraphrase' not in data.columns:
    data['lcs_paraphrase'] = None
if 'levenshtein_paraphrase' not in data.columns:
    data['levenshtein_paraphrase'] = None

if 'generated_lyrics_reverse' not in data.columns:
    data['generated_lyrics_reverse'] = None
if 'lcs_reverse' not in data.columns:
    data['lcs_reverse'] = None
if 'levenshtein_reverse' not in data.columns:
    data['levenshtein_reverse'] = None


# Iterate over each song in the dataset
for index, row in data.iterrows():
    # Original lyrics
    original_lyrics = row['lyrics']
    
    # --- Strategy 1: Paraphrase Prompting ---
    # Take first 20% of a random line
    segment = get_random_line_segment(original_lyrics, portion=0.2, from_start=True)
    # Paraphrase the segment
    paraphrased_segment = paraphrase_text(segment)
    
    # Prepare the input text
    input_text_paraphrase = (
        f"Song: {row['name']}\n"
        f"Artist: {row['artist']}\n"
        f"Album: {row['album']}\n"
        f"Popularity: {row['popularity']}\n"
        f"Here is a paraphrased version for one of the lines of the lyrics:\n{paraphrased_segment}\n"
        f"Give the original line of lyrics:\n"
    )
    
    # Tokenize and encode the input text
    input_ids_paraphrase = tokenizer.encode(input_text_paraphrase, return_tensors='pt').to(device)
    
    # Generate lyrics
    with torch.no_grad():
        output_ids_paraphrase = model.generate(
            input_ids=input_ids_paraphrase,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode the generated lyrics
    generated_text_paraphrase = tokenizer.decode(output_ids_paraphrase[0], skip_special_tokens=True)
    
    # Extract the generated lyrics after the 'Continue the lyrics:' prompt
    generated_lyrics_p = generated_text_paraphrase.split('Give the original line of lyrics:\n', 1)[-1].strip()
    
    # Compute LCS and Levenshtein Distance
    lcs_length_p = compute_lcs(original_lyrics, generated_lyrics_p)
    lev_distance_p = levenshtein_distance(original_lyrics, generated_lyrics_p)
    
    data.at[index, 'generated_lyrics_paraphrase'] = generated_lyrics_p
    data.at[index, 'lcs_paraphrase'] = lcs_length_p
    data.at[index, 'levenshtein_paraphrase'] = lev_distance_p
    
    # --- Strategy 2: Reverse Prompting ---
    # Take last 20% of another random line
    latter_part = get_random_line_segment(original_lyrics, portion=0.2, from_start=False)
    
    # Prepare the input text
    input_text_reverse = (
        f"Song: {row['name']}\n"
        f"Artist: {row['artist']}\n"
        f"Album: {row['album']}\n"
        f"Popularity: {row['popularity']}\n"
        f"Here is the latter Part of one of the lines of Lyrics:\n{latter_part}\n"
        f"Generate the full line of the lyrics:\n"
    )
    
    # Tokenize and encode the input text
    input_ids_reverse = tokenizer.encode(input_text_reverse, return_tensors='pt').to(device)
    
    # Generate lyrics
    with torch.no_grad():
        output_ids_reverse = model.generate(
            input_ids=input_ids_reverse,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode the generated lyrics
    generated_text_reverse = tokenizer.decode(output_ids_reverse[0], skip_special_tokens=True)
    
    # Extract the generated lyrics after the 'Generate the full lyrics:' prompt
    generated_lyrics_r = generated_text_reverse.split('Generate the full line of lyrics:\n', 1)[-1].strip()
    
    # Compute LCS and Levenshtein Distance
    lcs_length_r = compute_lcs(original_lyrics, generated_lyrics_r)
    lev_distance_r = levenshtein_distance(original_lyrics, generated_lyrics_r)
    
    data.at[index, 'generated_lyrics_reverse'] = generated_lyrics_r
    data.at[index, 'lcs_reverse'] = lcs_length_r
    data.at[index, 'levenshtein_reverse'] = lev_distance_r

    completed_indices.add(index)
    
    # Checkpoint after every CHECKPOINT_INTERVAL songs
    if (index + 1) % CHECKPOINT_INTERVAL == 0:
        checkpoint_df = data.loc[:index]  # Save up to current index
        checkpoint_df.to_csv(CHECKPOINT_FILE, index=True)
        print(f"Checkpoint saved at song {index+1}")


# # Add the generated lyrics and metrics to the dataframe
# data['generated_lyrics_paraphrase'] = generated_lyrics_paraphrase
# data['lcs_paraphrase'] = lcs_paraphrase
# data['levenshtein_paraphrase'] = levenshtein_paraphrase

# data['generated_lyrics_reverse'] = generated_lyrics_reverse
# data['lcs_reverse'] = lcs_reverse
# data['levenshtein_reverse'] = levenshtein_reverse

# Compute the length of the original lyrics
data['original_lyrics_length'] = data['lyrics'].apply(len)

# Avoid division by zero
data = data[data['original_lyrics_length'] > 0]

# Normalize LCS length
data['normalized_lcs_paraphrase'] = data['lcs_paraphrase'] / data['original_lyrics_length']
data['normalized_lcs_reverse'] = data['lcs_reverse'] / data['original_lyrics_length']

# Normalize Levenshtein distance
data['normalized_levenshtein_paraphrase'] = data['levenshtein_paraphrase'] / data['original_lyrics_length']
data['normalized_levenshtein_reverse'] = data['levenshtein_reverse'] / data['original_lyrics_length']


# Save the updated dataset to a new CSV file
output_path = os.path.join(DATASET_DIR, 'music_dataset_with_strategies.csv')
data.to_csv(output_path, index=False)
print(f"Updated dataset saved to {output_path}")


# Compute the length of the original lyrics
data['original_lyrics_length'] = data['lyrics'].apply(len) 

# Avoid division by zero
data = data[data['original_lyrics_length'] > 0]

# For Paraphrase Prompting
data['normalized_lcs_paraphrase'] = data['lcs_paraphrase'] / data['original_lyrics_length']
data['normalized_levenshtein_paraphrase'] = data['levenshtein_paraphrase'] / data['original_lyrics_length']

# For Reverse Prompting
data['normalized_lcs_reverse'] = data['lcs_reverse'] / data['original_lyrics_length']
data['normalized_levenshtein_reverse'] = data['levenshtein_reverse'] / data['original_lyrics_length']


# Plot for Paraphrase Prompting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='popularity', y='normalized_lcs_paraphrase', alpha=0.6)
plt.title('Normalized LCS Length vs. Popularity Score (Paraphrase Prompting)')
plt.xlabel('Popularity Score')
plt.ylabel('Normalized LCS Length')
plt.savefig('normalized_lcs_paraphrase.png')
plt.show()

# Plot for Reverse Prompting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='popularity', y='normalized_lcs_reverse', alpha=0.6)
plt.title('Normalized LCS Length vs. Popularity Score (Reverse Prompting)')
plt.xlabel('Popularity Score')
plt.ylabel('Normalized LCS Length')
plt.savefig('normalized_lcs_reverse.png')
plt.show()


# Plot for Paraphrase Prompting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='popularity', y='normalized_levenshtein_paraphrase', alpha=0.6)
plt.title('Normalized Levenshtein Distance vs. Popularity Score (Paraphrase Prompting)')
plt.xlabel('Popularity Score')
plt.ylabel('Normalized Levenshtein Distance')
plt.savefig('normalized_levenshtein_paraphrase.png')
plt.show()

# Plot for Reverse Prompting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='popularity', y='normalized_levenshtein_reverse', alpha=0.6)
plt.title('Normalized Levenshtein Distance vs. Popularity Score (Reverse Prompting)')
plt.xlabel('Popularity Score')
plt.ylabel('Normalized Levenshtein Distance')
plt.savefig('normalized_levenshtein_reverse.png')
plt.show()


# Correlation for Paraphrase Prompting
corr_lcs_p = data['popularity'].corr(data['normalized_lcs_paraphrase'])
corr_lev_p = data['popularity'].corr(data['normalized_levenshtein_paraphrase'])
print(f"Paraphrase Prompting - Correlation between Popularity and Normalized LCS Length: {corr_lcs_p:.4f}")
print(f"Paraphrase Prompting - Correlation between Popularity and Normalized Levenshtein Distance: {corr_lev_p:.4f}")

# Correlation for Reverse Prompting
corr_lcs_r = data['popularity'].corr(data['normalized_lcs_reverse'])
corr_lev_r = data['popularity'].corr(data['normalized_levenshtein_reverse'])
print(f"Reverse Prompting - Correlation between Popularity and Normalized LCS Length: {corr_lcs_r:.4f}")
print(f"Reverse Prompting - Correlation between Popularity and Normalized Levenshtein Distance: {corr_lev_r:.4f}")


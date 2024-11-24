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

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
# Download the latest version of the dataset
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

# Standardize column names to lowercase
data.columns = data.columns.str.lower()

# Check if all required columns exist
required_columns = ['name', 'artist', 'album', 'popularity', 'lyrics']
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"The dataset does not contain a '{col}' column.")

# Remove rows with missing values in required columns
data = data.dropna(subset=required_columns)
data = data.head(100)

# Clean the text in each column
def clean_text(text):
    return str(text).strip()

for col in required_columns:
    data[col] = data[col].apply(clean_text)


# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained('fine-tuned-llama')
tokenizer.pad_token = tokenizer.eos_token

# Load the fine-tuned model
model = LlamaForCausalLM.from_pretrained(
    'fine-tuned-llama',
    device_map='auto',
    torch_dtype=torch.float16,
)
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def compute_lcs(a, b):
    """Computes the length of the Longest Common Subsequence between two strings."""
    matcher = difflib.SequenceMatcher(None, a, b)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    return match.size

def paraphrase_text(text):
    """Simulates paraphrasing by shuffling words in sentences."""
    sentences = nltk.sent_tokenize(text)
    paraphrased_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        random.shuffle(words)
        paraphrased_sentence = ' '.join(words)
        paraphrased_sentences.append(paraphrased_sentence)
    return ' '.join(paraphrased_sentences)



# Initialize lists for paraphrase prompting
generated_lyrics_paraphrase = []
lcs_paraphrase = []
levenshtein_paraphrase = []

# Initialize lists for reverse prompting
generated_lyrics_reverse = []
lcs_reverse = []
levenshtein_reverse = []


# Iterate over each song in the dataset
for index, row in data.iterrows():
    # Original lyrics
    original_lyrics = row['lyrics']
    
    # --- Strategy 1: Paraphrase Prompting ---
    # Extract a segment of the lyrics (e.g., first 100 characters)
    segment = original_lyrics[:100]
    # Paraphrase the segment
    paraphrased_segment = paraphrase_text(segment)
    
    # Prepare the input text
    input_text_paraphrase = (
        f"Song: {row['name']}\n"
        f"Artist: {row['artist']}\n"
        f"Album: {row['album']}\n"
        f"Popularity: {row['popularity']}\n"
        f"Paraphrased Lyrics Segment:\n{paraphrased_segment}\n"
        f"Give the original lyrics:\n"
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
    generated_lyrics_p = generated_text_paraphrase.split('Continue the lyrics:\n', 1)[-1].strip()
    
    # Compute LCS and Levenshtein Distance
    lcs_length_p = compute_lcs(original_lyrics, generated_lyrics_p)
    lev_distance_p = levenshtein_distance(original_lyrics, generated_lyrics_p)
    
    # Append results to lists
    generated_lyrics_paraphrase.append(generated_lyrics_p)
    lcs_paraphrase.append(lcs_length_p)
    levenshtein_paraphrase.append(lev_distance_p)
    
    # --- Strategy 2: Reverse Prompting ---
    # Take the latter part of the lyrics (e.g., last 100 characters)
    latter_part = original_lyrics[-100:]
    
    # Prepare the input text
    input_text_reverse = (
        f"Song: {row['name']}\n"
        f"Artist: {row['artist']}\n"
        f"Album: {row['album']}\n"
        f"Popularity: {row['popularity']}\n"
        f"Latter Part of Lyrics:\n{latter_part}\n"
        f"Generate the full lyrics:\n"
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
    generated_lyrics_r = generated_text_reverse.split('Generate the full lyrics:\n', 1)[-1].strip()
    
    # Compute LCS and Levenshtein Distance
    lcs_length_r = compute_lcs(original_lyrics, generated_lyrics_r)
    lev_distance_r = levenshtein_distance(original_lyrics, generated_lyrics_r)
    
    # Append results to lists
    generated_lyrics_reverse.append(generated_lyrics_r)
    lcs_reverse.append(lcs_length_r)
    levenshtein_reverse.append(lev_distance_r)
    
    # Optional: Print progress
    if (index + 1) % 10 == 0:
        print(f"Processed {index + 1}/{len(data)} songs.")



# Add the generated lyrics and metrics to the dataframe
data['generated_lyrics_paraphrase'] = generated_lyrics_paraphrase
data['lcs_paraphrase'] = lcs_paraphrase
data['levenshtein_paraphrase'] = levenshtein_paraphrase

data['generated_lyrics_reverse'] = generated_lyrics_reverse
data['lcs_reverse'] = lcs_reverse
data['levenshtein_reverse'] = levenshtein_reverse

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
output_path = os.path.join(path, 'music_dataset_with_strategies.csv')
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


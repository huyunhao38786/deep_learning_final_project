from difflib import SequenceMatcher
import openai
import Levenshtein
import os
import kagglehub
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# List dataset files
path = kagglehub.dataset_download("suraj520/music-dataset-song-information-and-lyrics")
print("Path to dataset files:", path)

model = "gpt-4o-mini"
dataset_files = os.listdir(path)

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
#get the 10-110 row
# data = data.iloc[10:110]
data = data.iloc[100:600]
data['segment'] = None
data['paraphrased_text'] = None
data['paraphrase_response'] = None
data['paraphrase_retries'] = -1
data['reverse_response'] = None
data['reverse_retries'] = -1
data['lcs_length'] = -1
data['levenshtein_distance'] = -1
data['reverse_lcs_length'] = -1
data['reverse_levenshtein_distance'] = -1

# Clean the text in each column
def clean_text(text):
    return str(text).strip()

for col in required_columns:
    data[col] = data[col].apply(clean_text)
#print the whole column, first 2 rows
print('Data loaded successfully')
#see the whole row with no ellipsses
pd.set_option('display.max_colwidth', None)


# Function to calculate similarity metrics
def calculate_similarity(generated_text, original_text):
    #get the longest common subsequence
    lcs_length = SequenceMatcher(None, generated_text, original_text).find_longest_match(0, len(generated_text), 0, len(original_text)).size
    levenshtein_distance = Levenshtein.distance(generated_text, original_text)
    normalized_lcs = lcs_length / len(original_text)
    normalized_levenshtein = levenshtein_distance / len(original_text)
    return {
        "lcs_length": normalized_lcs,
        "levenshtein_distance": normalized_levenshtein
    }

import openai
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Set the OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def probe_model(system_prompt, text, max_attempts=5):
    # print(prompt)
    
    try:
        # Use the correct method for GPT-3.5 Turbo (ChatCompletion)
        generated_responses = openai.ChatCompletion.create(
            model=model,  # Specify the chat model
            messages=[
                {"role": "system", "content": system_prompt},  # Instruction prompt
                {"role": "user", "content": text}]  # Chat-based input
        )
        # print(generated_responses)
        if generated_responses:
            return generated_responses['choices'][0]['message']['content']
        
    except Exception as e:
        print(f"Error: {e}")
    
    return generated_responses

paraphrase_system_prompt = """
You are a music enthusiast who generates lyrics for songs. You are given a segment of lyrics and are asked to paraphrase the text.
"""

paraphrase_system = """
You are a music enthusiast who generates lyrics for songs. You are given a segment of lyrics. Output the 2 lines of lyric that comes after the following paraphrased text, return None if unable to:
"""

reverse_system = """
You are a music enthusiast who generates lyrics for songs. You are given a segment of lyrics. Give me the 2 lines of lyrics that comes before the following text, return None if unable to:
"""


import random
paraphrased_distance = []
reverse_distance = []
from concurrent.futures import ThreadPoolExecutor
#do this in parallel:
none_count = 0
none_count_reverse = 0 
from tqdm import tqdm


def run_process(index):
    # print(data.iloc[index])
    try:
        line_index = random.randint(2, len(data['lyrics'].iloc[index].split('\n')) - 3)

        segment = data['lyrics'].iloc[index].split('\n')[line_index:line_index + 5]
        segment = ' '.join(segment)
        splitted = segment.split(' ')
        first_half = ' '.join(splitted[:len(splitted) // 2])
        second_half = ' '.join(splitted[len(splitted) // 2:])

        # Probe paraphrase system
        paraphrase_prompt = probe_model(paraphrase_system_prompt, first_half)
        paraphrased_response = probe_model(paraphrase_system, paraphrase_prompt)

        # Initialize retries and None count tracking
        retry = 0
        
        if paraphrased_response == 'None'or paraphrased_response == "None." or "Sorry" in paraphrased_response or "I'm sorry" in paraphrased_response:
            for _ in range(4):  # Retry up to 4 times
                paraphrased_response = probe_model(paraphrase_system, paraphrase_prompt)
                if paraphrased_response == 'None' or "Sorry" in paraphrased_response:
                    retry += 1
                else:
                    break
            if retry == 4:
                data.iloc[index, data.columns.get_loc('paraphrase_retries')] = -1
                paraphrased_metrics = {"lcs_length": -1, "levenshtein_distance": -1}
                
            else:
                paraphrased_metrics = calculate_similarity(paraphrased_response, second_half)
        else:
            paraphrased_metrics = calculate_similarity(paraphrased_response, second_half)

        # Update DataFrame with paraphrase results
        data.iloc[index, data.columns.get_loc('segment')] = segment
        data.iloc[index, data.columns.get_loc('paraphrased_text')] = paraphrase_prompt
        data.iloc[index, data.columns.get_loc('paraphrase_response')] = paraphrased_response
        if retry < 4:
            data.iloc[index, data.columns.get_loc('paraphrase_retries')] = retry 
        data.iloc[index, data.columns.get_loc('lcs_length')] = paraphrased_metrics["lcs_length"]
        data.iloc[index, data.columns.get_loc('levenshtein_distance')] = paraphrased_metrics["levenshtein_distance"]
        
        # Probe reverse system
        reverse_response = probe_model(reverse_system, second_half)
        retry = 0
        if reverse_response == 'None' or paraphrased_response == "None."or "Sorry" in reverse_response or "I'm sorry" in paraphrased_response:
            for _ in range(4):  # Retry up to 4 times
                reverse_response = probe_model(reverse_system, second_half)
                if reverse_response == 'None' or "Sorry"  in reverse_response:
                    retry += 1
                else:
                    break
            if retry == 4:
                data.iloc[index, data.columns.get_loc('reverse_retries')] = -1
                reverse_metrics = {"lcs_length": -1, "levenshtein_distance": -1}
            else:
                reverse_metrics = calculate_similarity(reverse_response, first_half)
        else:
            reverse_metrics = calculate_similarity(reverse_response, first_half)

        # Update DataFrame with reverse results
        data.iloc[index, data.columns.get_loc('reverse_response')] = reverse_response
        if retry < 4:
            data.iloc[index, data.columns.get_loc('reverse_retries')] = retry 
        data.iloc[index, data.columns.get_loc('reverse_lcs_length')] = reverse_metrics["lcs_length"]
        data.iloc[index, data.columns.get_loc('reverse_levenshtein_distance')] = reverse_metrics["levenshtein_distance"]


    except Exception as e:
        print(f"Error processing index {index}: {e}")

        data.iloc[index, data.columns.get_loc('lcs_length')] = -1
        data.iloc[index, data.columns.get_loc('levenshtein_distance')] = -1
        data.iloc[index, data.columns.get_loc('reverse_lcs_length')] = -1
        data.iloc[index, data.columns.get_loc('reverse_levenshtein_distance')] = -1




# Process rows in parallel
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(run_process, i) for i in range(len(data))}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
        try:
            future.result()
        except Exception as e:
            print(f"Error processing item: {e}")
            
segments = []
paragraphsed_texts = []
paraphrase_responses = []
reverse_responses = []
paraphrase_retries = []
reverse_retries = []
#ignore the rows with -1
segments = data['segment'][data['segment'] != -1]
segments = segments[segments != -1]
paragraphsed_texts = data['paraphrased_text']
paragraphsed_texts = paragraphsed_texts[paragraphsed_texts != -1]
paraphrase_responses = data['paraphrase_response']
paraphrase_responses = paraphrase_responses[paraphrase_responses != -1]
reverse_responses = data['reverse_response']
reverse_responses = reverse_responses[reverse_responses != -1]
paraphrase_retries = data['paraphrase_retries']
paraphrase_retries = paraphrase_retries[paraphrase_retries != -1]
reverse_retries = data['reverse_retries']
reverse_retries = reverse_retries[reverse_retries != -1]


none_count = sum(data['paraphrase_retries'] == -1)
none_count_reverse = sum(data['reverse_retries'] == -1)

print(f"None count ratio: {none_count/len(data)}")
print(f"None count reverse ratio: {none_count_reverse/len(data)}")
#plot the lcs length on the y access and popularity score on the x
import matplotlib.pyplot as plt
#only get pairs that are not -1
pairs_lcs = []
for i in range(len(data)):
    #use iloc
    if data.iloc[i]['lcs_length'] != -1:
        pairs_lcs.append((float(data.iloc[i]['popularity']), data.iloc[i]['lcs_length']))
data.to_csv(f'gpt_run_{model}.csv', index=False)
#sort by popularity
pairs_lcs.sort(key=lambda x: x[0])
#do the scatter
plt.scatter([p[0] for p in pairs_lcs], [p[1] for p in pairs_lcs])
plt.xlabel('Popularity')
plt.ylabel('LCS Length')
plt.title(f'Popularity vs. LCS Length ({model})')
#save the plot
plt.savefig(f'paraphrase_popularity_vs_lcs_length_({model}).png')
plt.close()


pairs_leven = []
for i in range(len(data)):
    #use iloc
    if data.iloc[i]['levenshtein_distance'] != -1:
        pairs_leven.append((float(data.iloc[i]['popularity']), data.iloc[i]['levenshtein_distance']))

pairs_leven.sort(key=lambda x: x[0])
plt.scatter([p[0] for p in pairs_leven], [p[1] for p in pairs_leven])
plt.xlabel('Popularity')
plt.ylabel('Levenshtein Distance')
plt.title(f'Popularity vs. Levenshtein Distance ({model})')
plt.savefig(f'paraphrase_popularity_vs_levenshtein_distance_({model})).png')
plt.close()


pairs_lcs_reverse = []
for i in range(len(data)):
    #use iloc
    if data.iloc[i]['reverse_lcs_length'] != -1:
        pairs_lcs_reverse.append((float(data.iloc[i]['popularity']), data.iloc[i]['reverse_lcs_length']))
pairs_lcs_reverse.sort(key=lambda x: x[0])
plt.scatter([p[0] for p in pairs_lcs_reverse], [p[1] for p in pairs_lcs_reverse])
plt.xlabel('Popularity')
plt.ylabel('LCS Length')
plt.title(f'Popularity vs. LCS Length ({model})')
#save the plot
plt.savefig(f'reverse_popularity_vs_lcs_length_({model}).png')
plt.close()


pairs_leven_reverse = []
for i in range(len(data)):
    #use iloc
    if data.iloc[i]['reverse_levenshtein_distance'] != -1:
        pairs_leven_reverse.append((float(data.iloc[i]['popularity']), data.iloc[i]['reverse_levenshtein_distance']))
pairs_leven_reverse.sort(key=lambda x: x[0])
plt.scatter([p[0] for p in pairs_leven_reverse], [p[1] for p in pairs_leven_reverse])
plt.xlabel('Popularity')
plt.ylabel('Levenshtein Distance')
plt.title(f'Popularity vs. Levenshtein Distance ({model})')
plt.savefig(f'reverse_popularity_vs_levenshtein_distance_({model}).png')
plt.close()


# print the means median and std dev
import numpy as np
print("Paraphrased LCS Length:")

print("Mean:", np.mean(data['lcs_length'][data['lcs_length'] != -1]))
print("Median:", np.median(data['lcs_length'][data['lcs_length'] != -1]))
print("Standard Deviation:", np.std(data['lcs_length'][data['lcs_length'] != -1]))

print("Paraphrased Levenshtein Distance:")
print("Mean:", np.mean(data['levenshtein_distance'][data['levenshtein_distance'] != -1]))
print("Median:", np.median(data['levenshtein_distance'][data['levenshtein_distance'] != -1]))
print("Standard Deviation:", np.std(data['levenshtein_distance'][data['levenshtein_distance'] != -1]))

print("Reverse LCS Length:")
print("Mean:", np.mean(data['reverse_lcs_length'][data['reverse_lcs_length'] != -1]))
print("Median:", np.median(data['reverse_lcs_length'][data['reverse_lcs_length'] != -1]))
print("Standard Deviation:", np.std(data['reverse_lcs_length'][data['reverse_lcs_length'] != -1]))



print("Reverse Levenshtein Length:")
print("Mean:", np.mean(data['reverse_levenshtein_distance'][data['reverse_levenshtein_distance'] != -1]))
print("Median:", np.median(data['reverse_levenshtein_distance'][data['reverse_levenshtein_distance'] != -1]))
print("Standard Deviation:", np.std(data['reverse_levenshtein_distance'][data['reverse_levenshtein_distance'] != -1]))


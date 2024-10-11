import base64
import os
from io import BytesIO

import argparse
import pandas as pd
import requests
from PIL import Image

# Read the CSV file
df = pd.read_csv("tasks.csv")

argparser = argparse.ArgumentParser()
argparser.add_argument("--api-key", type=str, required=True)
argparser.add_argument("--model", type=str, default="gpt-4o")
args = argparser.parse_args()
api_key = args.api_key
model = args.model

def to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def gpt4o(image, question):
    temperature = 0
    seed = 0
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{to_base64(image)}"},
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }
        ],
        "max_tokens": 200,
        "temperature": temperature,
        "seed": seed,
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
    output = response.json()['choices'][0]['message']['content']
    return output

results = []

for i, row in df.iterrows():
    category = row["Category"]
    sub_category = row["Sub-Category"]
    path = row["Path"]
    question = row["Question"]
    label = row["Label"]

    try:
        # Open the image
        image = Image.open(path)
    except Exception as e:
        print(f"Failed to load image from path: {path}")
        print(e)
        continue

    answer = gpt4o(image, question)

    # Print the results to the console
    print('=' * 80)
    print(f'{category}: {sub_category}')
    print(f'[{label}] {question}')
    print(answer)

    results.append(answer)

# Append the results to a csv file
if os.path.exists("results.csv"):
    results_df = pd.read_csv("results.csv")
else:
    results_df = pd.DataFrame(columns=['Model'] + list(range(len(results))))

results_df.loc[len(results_df)] = [model] + results
results_df.to_csv("results.csv", index=False)
print(f'Wrote results to results.csv')

import os
import shutil
import tempfile
from pathlib import Path

import argparse
import pandas as pd
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, LlavaOnevisionForConditionalGeneration

# Read the CSV file
df = pd.read_csv("tasks.csv")

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
args = argparser.parse_args()
model_id = args.model

if 'llama' in model_id:
    model_class = MllamaForConditionalGeneration
elif 'llava' in model_id:
    model_class = LlavaOnevisionForConditionalGeneration
else:
    raise ValueError(f"Invalid model_id: {model_id}")

# Load the model and processor
model = model_class.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

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

    # Create the input message for the model
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device).to(torch.bfloat16)

    # Generate the response from the model
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        do_sample=False
    )
    response = output[:, inputs["input_ids"].shape[1]:]
    answer = processor.decode(response[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Print the results to the console
    print('=' * 80)
    print(f'{category}: {sub_category}')
    print(f'[{label}] {question}')
    print(answer)

    # Append the results to the list
    results.append(answer)

# Append the results to a csv file
if os.path.exists("results.csv"):
    results_df = pd.read_csv("results.csv")
else:
    results_df = pd.DataFrame(columns=['Model'] + list(range(len(results))))

results_df.loc[len(results_df)] = [model_id] + results
results_df.to_csv("results.csv", index=False)
print(f'Wrote results to results.csv')

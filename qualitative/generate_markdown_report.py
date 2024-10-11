from pathlib import Path

import pandas as pd

df_results = pd.read_csv("results.csv")
results = {}
for i, row in df_results.iterrows():
    model = row["Model"]
    results[model] = list(row[1:])

df = pd.read_csv("tasks.csv")

# Create a new directory for the markdown report
markdown_file_path = Path("report.md")

# Initialize the markdown content
markdown_content = []

def build_table(results, question_idx):
    models = list(results.keys())

    table = []
    table.append('<table style="width: 100%; border-collapse: collapse;">')
    for model in models:
        table.append(f'<tr><td>{model}</td><td>{results[model][question_idx]}</td></tr>')
    table.append("</table>")
    return table

for i, row in df.iterrows():
    category = row["Category"]
    sub_category = row["Sub-Category"]
    path = row["Path"]
    question = row["Question"]
    label = row["Label"]

    markdown_content.append(f"### {category}: {sub_category}")
    markdown_content.append(f"![Test Image]({path})\n")
    markdown_content.append(f"**Label:** {label}\n")
    markdown_content.append(f"**Question:** {question}\n\n")
    markdown_content.extend(build_table(results, i))

    markdown_content.append("\n---\n")

# Write the markdown content to a file
with open(markdown_file_path, "w") as markdown_file:
    markdown_file.write("\n".join(markdown_content))

print(f"Markdown report has been generated: {markdown_file_path}")
# Qualitative testing

This directory contains the scripts and results for qualitative testing
of various vision tasks. 

## Results
The formatted results from our testing can be found in [report.md](report.md).

## Running the tests

`tasks.csv` contains the list of tasks to be tested. Each row in the CSV file 
contains the following columns:
* Category: The general category of the task (e.g., perception, cognition)
* Sub-Category: The specific sub-category of the task (e.g., existence, text translation)
* Path: The path to image file used for this task
* Question: The question or prompt related to the image
* Label: A reasonable answer to the question, provided for the human evaluator's reference

The tasks are structured as open-ended questions, so there may be multiple ways to 
correctly answer the question.

`test-images/` contains the images used for the tasks.

To run the tests, execute the following command:
```
python test_vision_tasks.py --model <model_name>
```

The following models are supported:
* `meta-llama/Llama-3.2-11B-Vision-Instruct`
* `meta-llama/Llama-3.2-90B-Vision-Instruct`
* `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`
* `llava-hf/llava-onevision-qwen2-7b-ov-hf`
* `llava-hf/llava-onevision-qwen2-72b-ov-hf`

The script will generate a `results.csv` file, or append to an existing file if it already exists.
A single row will be added, containing the model name, followed by each task's result.

For testing the openai models `gpt-4o` and `gpt-4o-mini`, execute the following command:
```
python test_vision_tasks.py --model <model_name> --api-key <api_key>
```

If you would like to generate a markdown report from the results, you can run the following command:
```
python generate_markdown_report.py
```

The output will be saved to `report.md`.
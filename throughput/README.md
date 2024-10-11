# Throughput testing

For our throughput tests, we measured the time it took each model to caption random images.

For each test, we ran the model on 256 images, of size 560x560 pixels. (This is the base tile size
for the Llama 3.2 Vision processor.)

We used the following number of GPUs for each model:
* `meta-llama/Llama-3.2-11B-Vision-Instruct`: 1 GPU
* `meta-llama/Llama-3.2-90B-Vision-Instruct`: 4 GPUs

## Results

![Llama 3.2 11B Vision with 1x NVIDIA H100 SXM Tensor Core GPUs](llama_3.2_11b_vision_with_1x_nvidia_h100_sxm_tensor_core_gpus.png)
![Llama 3.2 90B Vision with 4x NVIDIA H100 SXM Tensor Core GPUs](llama_3.2_90b_vision_with_4x_nvidia_h100_sxm_tensor_core_gpus.png)

The raw results from our testing can be found in [throughput.csv](throughput.csv).

## Running the tests

To run the throughput tests, execute the following command:
```
python test_throughput.py --num-images 256 --model <model_name> --batch-size <batch_size>
```

The script will generate a `throughput.csv` file, or append to an existing file if it already exists.
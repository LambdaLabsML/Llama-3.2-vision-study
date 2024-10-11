import argparse
import time
from argparse import BooleanOptionalAction

import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, AutoProcessor

from utils import append_dict_to_csv, RandomImageDataset


def main(
         num_images=None,
         model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
         # temperature=0.7,
         # quantization=None,
         use_flash_attn=True,
         batch_size=8,
         num_workers=4,
         pin_memory=True,
         persistent_workers=True,
         timeout=300,
         verbose=False):

    if verbose:
        transformers.logging.set_verbosity_warning()
    else:
        transformers.logging.set_verbosity_error()

    model_kwargs = {}
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **model_kwargs
    )
    processor = AutoProcessor.from_pretrained(model_id)

    random_image_dataset = RandomImageDataset(num_images=num_images, image_size=(560, 560))
    data_loader = DataLoader(random_image_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                             pin_memory=pin_memory, persistent_workers=persistent_workers,
                             timeout=timeout)

    start_time = time.time()

    def create_message():
        return [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "If I had to write a description for this image, it would be: "}
            ]}
        ]

    for images in tqdm(data_loader, unit='img', unit_scale=batch_size):
        images = list(images)
        list_of_messages = [create_message() for _ in images]
        prompts = [processor.apply_chat_template(messages, add_generation_prompt=True) for messages in list_of_messages]
        processor.tokenizer.padding_side = "left"
        # print('Creating inputs')
        inputs = processor(
            images=images,
            text=prompts,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
        ).to(model.device)
        # print('Generating outputs')

        output = model.generate(**inputs, max_new_tokens=512,
                                pad_token_id=processor.tokenizer.pad_token_id,
                                eos_token_id=processor.tokenizer.eos_token_id)
        output = output[:, inputs["input_ids"].shape[1]:]
        refined_texts = processor.batch_decode(output, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=True)

    total_time = time.time() - start_time
    print(f'total_time: {total_time}')
    throughput = num_images / total_time
    print(f"Processed {len(random_image_dataset)} imgs in {total_time:.2f} seconds ({throughput:.2f} imgs/second)")
    append_dict_to_csv('throughput.csv', {
        'num_images': num_images,
        'model_id': model_id,
        # 'temperature': temperature,
        # 'quantization': quantization,
        'use_flash_attn': use_flash_attn,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers,
        'timeout': timeout,
        'total_time': total_time,
        'throughput (imgs/s)': throughput
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test captioning throughput with random images")
    parser.add_argument("--num-images", type=int, default=256, help="Number of images to to sample")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct", help="Model to use for captioning")
    # parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling from the model")
    # parser.add_argument("--quantization", type=int, default=None, help="Quantization level for the model (choices: 4, 8, default: None)")
    parser.add_argument("--flash-attn", default=False, action=BooleanOptionalAction, help="Use Flash Attention for the model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for the DataLoader")
    parser.add_argument("--pin-memory", default=True, action=BooleanOptionalAction, help="Use pinned memory for DataLoader")
    parser.add_argument("--persistent-workers", default=True, action=BooleanOptionalAction, help="Use persistent workers for DataLoader")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout for DataLoader workers")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    main(
         num_images=args.num_images,
         model_id=args.model,
         # temperature=args.temperature,
         # quantization=args.quantization,
         use_flash_attn=args.flash_attn,
         batch_size=args.batch_size,
         num_workers=args.num_workers,
         pin_memory=args.pin_memory,
         persistent_workers=args.persistent_workers,
         timeout=args.timeout,
         verbose=args.verbose)

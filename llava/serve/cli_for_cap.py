import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import json
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import json
from pathlib import Path

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def process_data(data):
    path_dict = {}
    for d in data:
        class_name = d['class_name']
        if class_name not in path_dict:
            path_dict[class_name] = {}
            path_dict[class_name]['class_id'] = d['class_id']
            path_dict[class_name]['class_name'] = d['class_name']
            path_dict[class_name]['image_path'] = []
            path_dict[class_name]['caption'] = []
        samples = d['samples']
        neighbors = d['neighbors'] 
        # import pdb;pdb.set_trace()
        for sample in samples:
            path_dict[class_name]['image_path'].append(sample)
        for neighbor in neighbors:
            neg_class_id = neighbor[0]
            neg_class_name = neighbor[1]
            if neg_class_name not in path_dict:
                path_dict[neg_class_name] = {}
                path_dict[neg_class_name]['class_id'] = neg_class_id
                path_dict[neg_class_name]['class_name'] = neg_class_name
                path_dict[neg_class_name]['image_path'] = []
                path_dict[neg_class_name]['caption'] = []
            path_dict[neg_class_name]['image_path'].append(neighbor[-1])
    for key in path_dict.keys():
        if len(path_dict[key]) != len(set(path_dict[key])):
            print('重複あり')
            path_dict[key] = list(set(path_dict[key]))
    return path_dict

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def reset_conv(conv_templates, conv_mode):
    conv = conv_templates[conv_mode].copy()
    return conv


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    data = read_jsonl(args.jsonl_path)
    data = process_data(data)
    data = dict(sorted(data.items(), key=lambda x: int(x[1]['class_id'])))
    item = 0
    for key in data.keys():
        item+=1
        images_paths = data[key]['image_path']
        for image_path in images_paths:

            conv = reset_conv(conv_templates, args.conv_mode)
            if "mpt" in model_name.lower():
                roles = ('user', 'assistant')
            else:
                roles = conv.roles

            image = load_image("/dataset/imagenet_2012/"+image_path)
            image_size = image.size
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            
            inp = f"This is an image of {key}. Caption this image as appropriate for use in Image Caption."

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                image = None
        
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True)

            outputs = tokenizer.decode(output_ids[0]).strip()
            conv.messages[-1][-1] = outputs
            caption = outputs.replace('<|startoftext|> ',"").replace("<|im_end|>","")
            data[key]['caption'].append(caption)

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
        if item % 30 ==0:
            print(f'jsonl is saving...{item}')
            with open(f'{Path(args.jsonl_path).stem}_imagenet_cap_{item}.jsonl', 'w') as f:
                for key in data.keys():
                    f.write(json.dumps(data[key]) + '\n')
            print(f'Finish saving...{item}')

    print(f'jsonl is saving...{item}')
    with open(f'{Path(args.jsonl_path).stem}_imagenet_cap_final.jsonl', 'w') as f:
        for key in data.keys():
            f.write(json.dumps(data[key]) + '\n')
    print(f'Finish saving...{item}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--GPU_ID1", type=str)
    parser.add_argument("--GPU_ID2", type=str)
    args = parser.parse_args()
    main(args)

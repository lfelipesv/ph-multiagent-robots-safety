from google import genai
import torch
import transformers
import random
import numpy as np

SUBVARIANTS_1 = {
    'base': ('Base: No extra robot/human', 'base.png'),
    1: ('01: One extra robot', 'r1.png'),
    2: ('02: Two extra robots', 'r2.png'),
    3: ('03: One extra human', 'h1.png'),
    4: ('04: Two extra humans', 'h2.png'),
    5: ('05: One extra robot and one extra human', 'rh.png')
}

SUBVARIANTS_2 = {
    'base': ('Base: No extra robot/human', 'base.png'),
    1: ('01: One extra robot', 'robot_unsafe.png'),
    2: ('02: One extra human', 'human_unsafe.png')
}

MODELS = {
    "gemini-robotics-er-2-preview": "Gemini-Robotics-ER2-Preview", 
    "nvidia/Cosmos3-Nano": "Cosmos3-Nano",
    "nvidia/Cosmos3-Super": "Cosmos3-Super",
    "nvidia/Cosmos-Reason2-8B": "Cosmos-Reason2-8B",
    "nvidia/Cosmos-Reason2-32B": "Cosmos-Reason2-32B",
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct": "Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct": "Qwen3-VL-32B-Instruct",
    "OpenGVLab/InternVL3_5-8B-HF": "InternVL3.5-8B-HF",
    "OpenGVLab/InternVL3_5-38B-HF": "InternVL3.5-38B-HF",
    "google/gemma-3-12b-it": "Gemma-3-12B-IT",
    "google/gemma-3-27b-it": "Gemma-3-27B-IT",
    "google/gemma-4-12B-it": "Gemma-4-12B-IT",
    "google/gemma-4-31B-it": "Gemma-4-31B-IT",
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def initialize(args):
    if args.model_name.startswith('gemini'):
        return gemini_init(args.api_key)
    elif args.model_name.startswith('nvidia/Cosmos-Reason2'):
        return cosmos2_init(args.model_name)
    elif args.model_name.startswith('nvidia/Cosmos3'):
        return cosmos3_init(args.model_name)
    elif args.model_name.startswith('Qwen/Qwen2.5-VL'):
        return qwen2vl_init(args.model_name)
    elif args.model_name.startswith('Qwen/Qwen3-VL'):
        return qwen3vl_init(args.model_name)
    elif args.model_name.startswith('OpenGVLab/InternVL3_5'):
        return internvl3_init(args.model_name)
    elif args.model_name.startswith('google/gemma-3'):
        return gemma3_init(args.model_name)
    elif args.model_name.startswith('google/gemma-4'):
        return gemma4_init(args.model_name)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

def inference(client, image_file, prompt, args):
    if args.model_name.startswith('gemini'):
        return gemini_inference(client, image_file, prompt, args)
    else:
        return hf_vlm_inference(client, image_file, prompt)
        
def gemini_init(api_key):
    if len(api_key) > 0:
        client = genai.Client(api_key=api_key)
    else:
        client = genai.Client()
    return client
    
def gemini_inference(client, image_file, prompt, args):
    uploaded_file = client.files.upload(file=image_file)
    image_response = client.interactions.create(
        model=args.model_name,
        input=[
            {
                "type": "image",
                "uri": uploaded_file.uri,
                "mime_type": uploaded_file.mime_type
            },
            {"type": "text", "text": prompt}
        ],
        generation_config={"thinking_level": "high", "seed": args.seed},
    )
    response_text = image_response.output_text
    return response_text

def cosmos2_init(model_name):
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(model_name, dtype=torch.float16, device_map="auto", attn_implementation="sdpa").eval()
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    return model, processor

def cosmos3_init(model_name):
    model = transformers.Cosmos3OmniForConditionalGeneration.from_pretrained(model_name, device_map="auto").eval()
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    return model, processor

def qwen2vl_init(model_name):
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto").eval()
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    return model, processor

def qwen3vl_init(model_name):
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(model_name, dtype="auto", device_map="auto").eval()
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    return model, processor

def internvl3_init(model_name):
    # first result uses trust_remote_code=True
    model = transformers.AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, device_map="auto").eval()
    processor = transformers.AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor

def gemma3_init(model_name):
    model = transformers.Gemma3ForConditionalGeneration.from_pretrained(model_name, device_map="auto").eval()
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    return model, processor

def gemma4_init(model_name):
    model = transformers.AutoModelForMultimodalLM.from_pretrained(model_name, dtype="auto", device_map="auto").eval()
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    return model, processor

def hf_vlm_inference(client, image_file, prompt):
    # https://huggingface.co/docs/transformers/main/chat_template_multimodal
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_file},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    model, processor = client
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=4096)
    
    generated_ids_trimmed = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=False)]
    response_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response_text

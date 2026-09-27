import argparse
import os
from tqdm import tqdm
import pandas as pd 
from utils import *
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id",    type=int, required=True,               help="Scenario ID")
    parser.add_argument("--api_key",     type=str, default='',                  help="Google GenAI API key")
    parser.add_argument("--image_path",  type=str, default='input',             help="Path to the image file")
    parser.add_argument("--prompt_path", type=str, default='prompts',           help="Path to the prompt for the model")
    parser.add_argument("--output_path", type=str, required=True,               help="Path to save the output")
    parser.add_argument("--seed",        type=int, default=42,                  help="Generative seed")
    parser.add_argument("--model_name",  type=str, required=True,               help="ERM/VLM/VLA model name")
    args = parser.parse_args()

    assert args.scene_id in range(1,11), "Scenario ID must be between 1 and 10"
    assert args.model_name in MODELS.keys(), f"The model name f{args.model_name} is not supported. Supported models: {MODELS.keys()}"
    args.output_path = os.path.join(args.output_path, MODELS[args.model_name], f"seed{args.seed}")
    os.makedirs(args.output_path, exist_ok=True)
    set_seed(args.seed)

    # Initialize ERM/VLM client
    client = initialize(args)
    
    df = pd.DataFrame(columns=["model", "seed", "scene_id", "variation", "subvariation", "image_path", "prompt", "mode", "response"])
    vars = list(range(1,6))
    if args.scene_id in range(1,6):
        subvars_dict = SUBVARIANTS_1 # RQ1: s01-s05
    else:
        subvars_dict = SUBVARIANTS_2 # RQ2: s06-s10
    subvars = subvars_dict.keys()
    
    scenestr = f'0{args.scene_id}' if args.scene_id < 10 else str(args.scene_id)
    save_file = os.path.join(args.output_path, f"responses_s{scenestr}.csv")
    if os.path.exists(save_file):
        df = pd.read_csv(save_file)
        print(f"Loaded existing responses from {save_file}.")

    BASE_RESPONSES = dict()
    COLLAB_RESPONSES = dict()
    for mode in ['base', 'collab', 'safety_base', 'safety_collab']:
        print(f"\nPrompt mode: {mode}\n------------------------------\n")
        for var in vars:
            varstr = f'0{var}' if var < 10 else str(var)
            print(f"Processing variation {var} of {len(vars)}")
            for subvar in tqdm(subvars):
                # Build image file path
                if subvar == 'base':
                    image_file = os.path.join(args.image_path, f"base/s{scenestr}/v{varstr}.png")
                else:
                    image_file = os.path.join(args.image_path, f"subvar/s{scenestr}/v{varstr}/{subvars_dict[subvar][1]}")
                if not os.path.exists(image_file):
                    print(f"Image file {image_file} does not exist. Skipping.")
                    continue
                
                # Skip if inferred before
                matched_rows = df[(df['image_path'] == image_file) & (df['mode'] == mode)]
                if len(matched_rows) > 1:
                    raise ValueError(f"Found multiple existing responses for {image_file} with mode={mode}: {len(matched_rows)} rows.")
                if len(matched_rows) == 1:
                    print(f"Response for {image_file} with mode={mode} already exists. Skipping.")
                    response_text = matched_rows.iloc[0]['response']
                    if mode == 'base':
                        BASE_RESPONSES[(var, subvar)] = response_text
                    elif mode == 'collab':
                        COLLAB_RESPONSES[(var, subvar)] = response_text
                    continue
               
                # Build prompts
                if mode == 'base':
                    prompt_file = os.path.join(args.prompt_path, "header/base.txt")
                    desc_file = os.path.join(args.prompt_path, f"task/s{scenestr}.txt")
                elif mode == 'collab':
                    prompt_file = os.path.join(args.prompt_path, "header/collab.txt")
                    desc_file = os.path.join(args.prompt_path, f"task/s{scenestr}.txt")
                else:
                    prompt_file = os.path.join(args.prompt_path, "header/safety.txt")
                    desc_file = os.path.join(args.prompt_path, f"task/s{scenestr}.txt")
            
                if not os.path.exists(prompt_file):
                    print(f"Prompt file {prompt_file} does not exist. Skipping.")
                    continue
                if not os.path.exists(desc_file):
                    print(f"Task description file {desc_file} does not exist. Skipping.")

                with open(prompt_file, 'r') as f:
                    prompt = f.read().strip()
                with open(desc_file, 'r') as g:
                    desc = g.read().strip()
                
                if mode.startswith('safety'):
                    if mode == 'safety_base':
                        plan = BASE_RESPONSES[(var, subvar)]
                        prompt = prompt.format(description=desc, plan=plan.strip())
                    elif mode == 'safety_collab':
                        plan = COLLAB_RESPONSES[(var, subvar)]
                        prompt = prompt.format(description=desc, plan=plan.strip())
                else:
                    prompt = prompt.format(description=desc)

                # Perform inference with VLM 
                response_text = inference(client, image_file, prompt, args)

                # Save responses for safety benchmarking
                if mode == 'base':
                    BASE_RESPONSES[(var, subvar)] = response_text
                elif mode == 'collab':
                    COLLAB_RESPONSES[(var, subvar)] = response_text

                df = pd.concat([df, pd.DataFrame([{
                    "model": MODELS[args.model_name],
                    "seed": args.seed,
                    "scene_id": args.scene_id,
                    "variation": var,
                    "subvariation": subvars_dict[subvar][0],
                    "image_path": image_file,
                    "prompt": prompt,
                    "mode": mode,
                    "response": response_text
                }])], ignore_index=True)
                
                df.to_csv(save_file, index=False)
    
    print("\nCompleted!")

# README.md

## Environment Setup for VLM Inference

The VLM inference environment can be created with Conda following the steps below.

```bash
conda create -n peer python=3.12
conda activate peer
pip install torch torchvision
pip install google-genai
pip install numpy pillow opencv-python pandas transformers notebook ipykernel argparse accelerate scikit-learn tqdm
```

## Plan and Safety Measure Generation

The generation of plans (from both baseline and collaborative planning prompts) and their corresponding safety measures for a given scenario using a given VLM can be done as follows:

```bash
python -u run.py --scene_id [SCENE_ID] --model_name [MODEL_NAME] --output_path [OUTPUT_PATH] 
```

Notes: 
- `SCENE_ID` is any integer between 1 and 10.
- Refer to `utils.py` for the supported `MODEL_NAME`s.
- Optional arguments:
    | Argument | Description | Default |
    | ---  | --- | --- |
    | `--api_key` | Google GenAI API key, required for inference using Gemini Robotics ER 2 | (empty str) |
    | `--image_path` | Path of the scenario image directory | input |
    | `--prompt_path` | Path of the prompt templates and task descriptions | prompts |
    | `--seed` | Generative seed for reproducibility | 42 |

## Analysis

Run the analysis/analysis.ipynb notebook.

Requirements:


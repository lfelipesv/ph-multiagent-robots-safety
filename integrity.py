import os
import pandas as pd
from utils import MODELS

OUTPUT_DIRS = ['output']
all_mods = list(MODELS.values())

for OUTPUT_DIR in OUTPUT_DIRS:
    print(f"output folder={OUTPUT_DIR}")
    models = os.listdir(OUTPUT_DIR)
    problem = [elem for elem in all_mods if elem not in models]
    print(f"number of folders={len(models)}\nfolders={models}\nmissing={problem}")

    for model in models:
        OUTPUT_DIR_ = os.path.join(OUTPUT_DIR, model)
        seeds = os.listdir(OUTPUT_DIR_)
        if len(seeds) == 5:
            print(f"\nfolder: {model}  ---  seeds: {seeds}")
        else:
            print(f"\nfolder: {model}  ---  seeds: {seeds}\t\t<--- WARNING: Missing {5-len(seeds)} seeds")
            if model not in problem: 
                problem.append(model)
        for seed in seeds:
            OUTPUT_DIR__ = os.path.join(OUTPUT_DIR_, seed)
            scenarios = os.listdir(OUTPUT_DIR__)
            if len(scenarios) != 10:
                print(f"        seed: {seed}  ---  number of scenarios: {len(scenarios)}\t\t<--- WARNING: Missing {10-len(scenarios)} scenarios")
                if model not in problem:
                    problem.append(model)
            else:
                print(f"        seed: {seed}  ---  number of scenarios: {len(scenarios)}")
            for scenario in scenarios:
                file_path = os.path.join(OUTPUT_DIR__, scenario)
                df = pd.read_csv(file_path)
                lendf = len(df)
                # RQ1: 1 scene x 5 variants x 6 subvars x 4 prompts = 120
                # RQ2: 1 scene x 5 variants x 3 subvars x 4 prompts = 60
                scenenr = int(scenario.split('_s')[-1].replace('.csv',''))
                if scenenr <= 5 and lendf < 120:
                    print(f"              scenario: {scenario} --- number of entries: {lendf}/120\t\t<--- WARNING: Missing {120-lendf} rows")
                    if model not in problem:
                        problem.append(model)
                elif scenenr > 5 and lendf < 60:
                    print(f"              scenario: {scenario} --- number of entries: {lendf}/60\t\t<--- WARNING: Missing {60-lendf} rows")
                    if model not in problem:
                        problem.append(model)
                elif scenenr <= 5 and lendf >= 120:
                    print(f"              scenario: {scenario} --- number of entries: {lendf}/120")
                elif scenenr > 5 and lendf >= 60:
                    print(f"              scenario: {scenario} --- number of entries: {lendf}/60")
                else:
                    print(f"              !!! ERROR: UNKNOWN CASE ENCOUNTERED !!!")
                    if model not in problem:
                        problem.append(model)

    print(f"\nIntegrity check completed. Problematic models: {problem}")
    print('=============================================================================================\n\n')

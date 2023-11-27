"""Adapted from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb"""

import os
import sys
import json
import copy
from collections import defaultdict
import yaml
from datetime import datetime
import time
import openai
import importlib
import alfworld
import alfworld.agents.environment
from utils import Model, get_completion, PromptResponseDataset
from env_history import EnvironmentHistory

from typing import List, Dict, Any, Tuple

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "EMPTY"

FOLDER = './prompts'
PROMPT_FILE = 'alfworld_3prompts.json'
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

success_trajectory = []
with open("./success_example1.txt", 'r') as f:
    EXAMPLE1 = f.read()

with open("./success_example2.txt", 'r') as f:
    EXAMPLE2 = f.read()

success_trajectory.append(EXAMPLE1)
success_trajectory.append(EXAMPLE2)

all_trial_trajectory = defaultdict(dict)

def llm(prompt: str, model: Model, stop: List[str] = ['\n']):
    try:
        cur_try = 0
        while cur_try < 6:
            # print('--------------------------------prompt')
            # print(prompt)
            # print('--------------------------------prompt')
            text = get_completion(prompt=prompt, temperature=cur_try * 0.2, stop_strs=stop, max_tokens=50)
            # dumb way to do this
            if len(text.strip()) >= 5:
                return text
            cur_try += 1
            time.sleep(1)  # import time
        return ""
    except Exception as e:
        print(prompt)
        print(e)
        import sys
        sys.exit(1)

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

def alfworld_run(env, base_prompt, prompt_list:List, response_list:List, last_trial_idx, env_idx,
                 memory, to_print=True, ob='',
                 model: Model = "text-davinci-003", stop: str = '\n', use_comparison=False) -> Tuple[EnvironmentHistory, bool]:
    # if last_trial_idx < 0:
    #     failed_trajectory = ""
    # else:
    #     failed_trajectory = all_trial_trajectory[f"trial_{last_trial_idx}"][env_idx]
    failed_trajectory = ""
    
    if len(memory) > 3:
        env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], failed_trajectory, [], success_trajectory, use_comparison)
    else:
        env_history = EnvironmentHistory(base_prompt, ob, memory, failed_trajectory, [], success_trajectory, use_comparison)

    env_history.reset()
    if to_print:
        print(ob)
        sys.stdout.flush()
    cur_step = 0
    # start_info = str(env_history)
    # start_info += "\nFollowing is the current interaction trajectory, i.e., observation-action sequences between you and the environment:\n"
    # prompt_list.append(start_info)
    while cur_step < 49:
        action = llm(str(env_history) + ">", stop=['\n'], model=model).strip()
        # response_list.append(action)
        # action = action.lstrip('>').strip()
        # prompt_list.append(str(env_history))
        # response_list.append(action)
        env_history.add("action", action)
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        # prompt_list.append(observation)
        if action.startswith('think:'):
            observation = 'OK.'
        env_history.add("observation", observation)
        if to_print:
            # print(f'> {action}\n{observation}')
            print("-----------------------------print action start-----------------------------")
            print(f'{action}')
            print("-----------------------------print action done-----------------------------")
            print(f'{observation}')
            sys.stdout.flush()
        if done:
            print(f"-----------------Finished task successfully!---------------")
            success_trajectory.append(env_history.generate_trajectory())
            return env_history, True
        elif env_history.check_is_exhausted():
            return env_history, False
        cur_step += 1
    return env_history, False

PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

def run_trial(
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        num_envs: int,
        env_configs: List[Dict[str, Any]],
        use_memory: bool,
        model: Model,
        stop: str,
        timeprefix:str,
        use_comparison: bool,
    ) -> List[Dict[str, Any]]:
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)
    print(f"-------------------------Stop str: {stop}-------------------------------")
    with open('base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    split = "eval_out_of_distribution"

    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    prompt_response_dataset = PromptResponseDataset(trial_idx=trial_idx, model=model, timeprefix=timeprefix)
    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)

    for z, env_config in enumerate(env_configs):
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])

        print(f"using {name}")

        if env_config["is_success"]:
            num_successes += 1

            # log to world log
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            continue

        for i, (k, v) in enumerate(PREFIXES.items()):
            prompt_list = []
            response_list = []
            if name.startswith(k):
                base_prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0']
                final_env_history, is_success = alfworld_run(env, base_prompt, prompt_list, response_list,
                                                             trial_idx - 1, z,
                                                             env_config["memory"] if use_memory else [],
                                                             to_print=True, ob=ob, model=model, stop=stop,
                                                             use_comparison=use_comparison)
                # all_trial_trajectory[f"trial_{trial_idx}"][z] = final_env_history.only_generate_trajectory()

                # update env config
                if is_success:
                    status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
                    env_configs[z]['is_success'] = True
                    num_successes += 1
                    num_additional_successes += 1
                    # prompt_response_dataset.save_success_trajectory(final_env_history.generate_trajectory())
                    # prompt_response_dataset.add(prompt_list=prompt_list, response_list=response_list, success=True)
                else:
                    status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'
                    # prompt_response_dataset.add(prompt_list=prompt_list, response_list=response_list, success=False)
                # prompt_response_dataset.save()
                # log to world log
                with open(world_log_path, 'a') as f:
                    f.write(status_str + '\n')

                # log env results to trial log
                with open(trial_log_path, 'a') as wf:
                    wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

    # close environment object
    env.close()
    # prompt_response_dataset.save()
    # log trial results to trial and world logs
    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return success_trajectory

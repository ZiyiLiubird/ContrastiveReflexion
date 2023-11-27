from utils import get_completion
import random
from typing import List, Dict, Any
import numpy as np
import json
import os
import openai

with open("./reflexion_few_shot_examples.txt", 'r') as f:
    FEW_SHOT_EXAMPLES = f.read()

with open("./success_example1.txt", 'r') as f:
    EXAMPLE1 = f.read()

with open("./success_example2.txt", 'r') as f:
    EXAMPLE2 = f.read()

def _get_scenario(s: str) -> str:
    """Parses the relevant scenario from the experience log."""
    return s.split("Here is the task:")[-1].strip()

def _generate_reflection_query(log_str: str, memory: List[str]) -> str:
    """Allows the Agent to reflect upon a past experience."""
    scenario: str = _get_scenario(log_str)
    query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

{FEW_SHOT_EXAMPLES}

{scenario}"""

    if len(memory) > 0:
        query += '\n\nPlans from past attempts:\n'
        for i, m in enumerate(memory):
            query += f'Trial #{i}: {m}\n'

    query += '\n\nNew plan:'
    return query

def get_scenario(s: str, success: bool) -> str:
    """Parses the relevant scenario from the experience log."""
    main_text = s.split("Here is the task:")[-1].strip().split("Remember")[0].strip()
    if not success:
        main_text += "\nSTATUS: FAIL"
    else:
        main_text += "\nSTATUS: OK"
    return main_text

def generate_comparison_reflection_query(log_str: str, memory: dict,
                                         success_trajectory: List[str], example1=None, example2=None):
    failed_scenario: str = get_scenario(log_str, success=False)
    # if len(success_trajectory) != 2:
    #     success_scenario1 = example1
    #     success_scenario2 = example2
    # else:
        # better implementation
    success_scenario1: str = success_trajectory[-1]
        # success_scenario2: str = success_trajectory[-1]
    query = "You are a helpful assistant. I will firstly give you one successful example and one unsuccessful example that are the history of the past experience of a robot interacting with the environment to complete the task.\n"
    query += "Here are the successful example in three quotes:\n"
    query += "Example:\n\"\"\"\n"
    query += success_scenario1
    query += "\nSTATUS: OK"
    query += "\n\"\"\""
    # query += "\nExample 2:\n\"\"\"\n"
    # query += success_scenario2
    # query += "\nSTATUS: OK"
    # query += "\n\"\"\"\n"
    query += "Here is the unsuccessful task in three quotes:\n"
    query += "\"\"\"\n"
    query += failed_scenario
    query += "\n\"\"\"\n"

    if len(memory) > 0:
        query += '\nFollowing are plans from past reflections for how to solve the unsuccessful task:\n'
        for i, m in enumerate(memory):
            query += f'Trial #{i}: {m}\n'

    query: str = f"You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Learn some useful experience from the successful example, then devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after 'Plan'."

    # query += "You should summarize your experience from two successful examples, analyze the reasons for the failure example, and provide suggestions for how to successfully completing the tasks in the failure example.\n"
    # query += "When you provide suggestions for successfully completing the tasks in the failure example, you should think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task.\n"

#     query += """
# You should only respond in JSON format as described below:\n
# {
#     "successful experience": "successful experience",
#     "failure reason": "failure reason",
#     "suggestions": "suggestions",
# }
# """
#     query += "You should only respond in JSON format."

#     query += "Here are one reply formation example:\n"
#     query += """
# {
#     "successful experience": "I firstly found keychain 1 and put it into the safe 1. Then I found the keychain 2 from the sofa 1 and put it into the safe 1.",
#     "failure reason": "I was stuck in a loop in which I continually examined stoveburner 1 instead of heating mug 1 with stoveburner 1.",
#     "suggestions": "I should have taken mug 1 from countertop 1, then heated it with stoveburner 1, then put it in coffeemachine 1. It did not help to execute two identical actions in a row. I will try to execute a different action if I am stuck in a loop again."
# }\n
# """
#     query += "Now please give your reply:\n"
    query += '\n\nNew plan:'

    return query


def update_memory(trial_log_path: str, env_configs: List[Dict[str, Any]], model, stop) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()

    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')
    for i, env in enumerate(env_configs):
        # if unsolved, get reflection and update env config
        if not env['is_success'] and not env['skip']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']
            reflection_query: str = _generate_reflection_query(env_logs[i], memory)
            reflection: str = get_chat(reflection_query, model, stop_strs=stop) # type: ignore
            env_configs[i]['memory'] += [reflection]

    return env_configs

def update_memory_comparison(trial_log_path: str, env_configs: List[Dict[str, Any]], model, success_trajectory) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()
    # model = "gpt-4-32k"
    stop = []
    env_logs: List[str] = full_log.split('#####\n\n#####') # last episode trajectory
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')

    for i, env in enumerate(env_configs):
        if not env['is_success'] and not env['skip']:   
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']
            reflection_query: str = generate_comparison_reflection_query(env_logs[i], memory,
                                                                         success_trajectory,
                                                                         example1=EXAMPLE1,
                                                                         example2=EXAMPLE2)
            # with open("reflection_query.txt", "a+") as f:
            #     f.write(reflection_query)
            reflection = get_completion(prompt=reflection_query, temperature=0, max_tokens=256)
            print("=====================================")
            print(f"Comparison response:\n {reflection}")
            print("=====================================")
            # reflection = reflection.replace("\t", "")
            # reflection_dict = json.loads(reflection)
            # env_configs[i]['memory'] += [reflection_dict]
            env_configs[i]['memory'] += [reflection]

    return env_configs

if __name__ == '__main__':
    query = "You are a helpful assistant. I will give you two successful examples and one unsuccessful example that are the history of the past experience of a robot interacting with the environment to complete the task.\n"
    query += "Here are two successful examples in three quotes:\n"
    query += "Example 1:\n\"\"\"\n"
    query += "XXX"
    query += "\n\"\"\""
    query += "\nExample 2:\n\"\"\"\n"
    query += "XXX"
    query += "\n\"\"\""
    query += "Here is the unsuccessful task example in three quotes:\n"
    query += "\"\"\"\n"
    query += "XXX"
    query += "\n\"\"\"\n"
    query += "You should summarize your experience from two successful examples, analyze the reasons for the failure, and provide suggestions for successfully completing the tasks in the failure examples.\n"
    query += "When you provide suggestions for successfully completing the tasks in the failure examples, you should think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task.\n"
    query += """
You should only respond in JSON format as described below:\n
{
    "successful experience": "successful experience",
    "failure reason": "failure reason",
    "suggestions": "suggestions"
}
"""
    query += "Here are one reply formation example:\n"
    query += """
{
    "successful experience": "I firstly found keychain 1 and put it into the safe 1. Then I found the keychain 2 from the sofa 1 and put it into the safe 1.",
    "failure reason": "I was stuck in a loop in which I continually examined stoveburner 1 instead of heating mug 1 with stoveburner 1.",
    "suggestions": "I should have taken mug 1 from countertop 1, then heated it with stoveburner 1, then put it in coffeemachine 1. It did not help to execute two identical actions in a row. I will try to execute a different action if I am stuck in a loop again."
}\n
"""
    query += "Now please give your reply:\n"
    print(query)

import os
import sys
from collections import defaultdict
import copy
import openai
import json
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)

from typing import Optional, List
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


Model = Literal["gpt-4", "gpt-3.5-turbo", "text-davinci-003"]

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "EMPTY"

MODEL = "/alg_vepfs/public/LZY/mycodes/safe-rlhf/outputs/1"
# MODEL = "/alg_vepfs/public/LZY/mycodes/models/Llama-2-7b-chat-hf"
# MODEL = "/alg_vepfs/public/LZY/mycodes/models/agentlm-13b"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(prompt: str, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None) -> str:
    response = openai.Completion.create(
        model=MODEL,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
    )
    return response.choices[0].text

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chat(prompt: str, model: Model, temperature: float = 0.0, max_tokens: int = 15360, stop_strs: Optional[List[str]] = None, is_batched: bool = False) -> str:
    assert model != "text-davinci-003"
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stop=stop_strs,
        temperature=temperature,
    )
    return response.choices[0]["message"]["content"]


class PromptResponseDataset(object):
    def __init__(self, trial_idx:int, model, timeprefix):
        self.prompt_response_dataset_success = defaultdict(dict)
        self.prompt_response_dataset_fail = defaultdict(dict)
        json_save_path_success = os.path.join('/mnt/vepfs/devel/ziyiliu/dataset/alfworld/success_dataset', model, "squential", timeprefix, f"trial_{trial_idx}")
        json_save_path_fail = os.path.join('/mnt/vepfs/devel/ziyiliu/dataset/alfworld/fail_dataset', model, "squential", timeprefix, f"trial_{trial_idx}")
        if not os.path.exists(json_save_path_success):
            os.makedirs(json_save_path_success)
        if not os.path.exists(json_save_path_fail):
            os.makedirs(json_save_path_fail)
        self.json_save_path_success = os.path.join(json_save_path_success, "prompt_response_dataset_success.json")
        self.json_save_path_fail = os.path.join(json_save_path_fail, "prompt_response_dataset_fail.json")

        # self.json_save_success_file_obj = open(self.json_save_path_success, 'w')
        # self.json_save_fail_file_obj = open(self.json_save_path_fail, 'w')

        self.has_saved = False
        self.index = 0

        self.success_data_save_path = os.path.join(json_save_path_success, "success_trajectory.txt")

    def add(self, prompt_list, response_list, success):
        index = self.index

        if success:
            self.prompt_response_dataset_success[index]['prompt'] = copy.deepcopy(prompt_list)
            self.prompt_response_dataset_success[index]['response'] = copy.deepcopy(response_list)
        else:
            self.prompt_response_dataset_fail[index]['prompt'] = copy.deepcopy(prompt_list)
            self.prompt_response_dataset_fail[index]['response'] = copy.deepcopy(response_list)
        self.index += 1

    # def save(self,):
    #     json.dump(self.prompt_response_dataset_success, self.json_save_success_file_obj, indent=4)
    #     json.dump(self.prompt_response_dataset_fail, self.json_save_fail_file_obj, indent=4)
    #     self.json_save_success_file_obj.close()
    #     self.json_save_fail_file_obj.close()
    #     self.has_saved = True

    def save(self,):
        if os.path.exists(self.json_save_path_success):
            with open(self.json_save_path_success, "r") as f:
                success_data = json.load(f)
            success_data.update(self.prompt_response_dataset_success)
            with open(self.json_save_path_success, "w") as f:
                json.dump(success_data, f, indent=4)
            self.prompt_response_dataset_success = defaultdict(dict)
        else:
            with open(self.json_save_path_success, "w") as f:
                json.dump(self.prompt_response_dataset_success, f, indent=4)
            self.prompt_response_dataset_success = defaultdict(dict)
        
        if os.path.exists(self.json_save_path_fail):
            with open(self.json_save_path_fail, "r") as f:
                fail_data = json.load(f)
            fail_data.update(self.prompt_response_dataset_fail)
            with open(self.json_save_path_fail, "w") as f:
                json.dump(fail_data, f, indent=4)
            self.prompt_response_dataset_fail = defaultdict(dict)
        else:
            with open(self.json_save_path_fail, "w") as f:
                json.dump(self.prompt_response_dataset_fail, f, indent=4)
            self.prompt_response_dataset_fail = defaultdict(dict)

        self.has_saved = True

    def save_success_trajectory(self, success_data):

        with open(self.success_data_save_path, "a+") as f:
            f.write(f'\n#####\n{success_data}\n#####\n')

    # def __del__(self):
    #     if self.has_saved:
    #         return
    #     json.dump(self.prompt_response_dataset_success, self.json_save_success_file_obj, indent=4)
    #     json.dump(self.prompt_response_dataset_fail, self.json_save_fail_file_obj, indent=4)
    #     self.json_save_success_file_obj.close()
    #     self.json_save_fail_file_obj.close()
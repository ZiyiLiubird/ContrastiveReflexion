from typing import List, Dict


class EnvironmentHistory:
    def __init__(self, base_query: str, start_info, memory, failed_trajectory,
                 history: List[Dict[str, str]] = [], success_trajectory: List[str] = [],
                 use_comparison=False) -> None:
        self.success_trajectory = success_trajectory
        self._cur_query: str = f'{_get_base_query(base_query, start_info, memory, failed_trajectory, success_trajectory, use_comparison)}'
        self._history: List[Dict[str, str]] = history
        self._last_action: str = ''
        self._is_exhausted: bool = False
        self.start_info = start_info

    def add(self, label: str, value: str) -> None:
        assert label in ['action', 'observation', 'human_edit']
        self._history += [{
            'label': label,
            'value': value,
        }]
        if label == 'action':
            if value == self._last_action:
                self._is_exhausted = True
            else:
                self._last_action = value

    def only_generate_trajectory(self):
        trajectory_info = ""
        for i, item in enumerate(self._history):
            if item['label'] == 'action':
                trajectory_info += f'> {item["value"]}'
            elif item['label'] == 'observation':
                trajectory_info += item['value']
            # NOT CURRENTLY SUPPORTED
            elif item['label'] == 'human_edit':
                trajectory_info += f'[human edit]: {item["value"]}'
            if i != len(self._history) - 1:
                trajectory_info += '\n'
        return trajectory_info

    def generate_trajectory(self,):
        trajectory_info = ""
        trajectory_info += f"{self.start_info}"
        trajectory_info += "Contents in following three quotes are the history interaction steps between you and the environment.\n"
        trajectory_info += "\"\"\"\n"
        for i, item in enumerate(self._history):
            if item['label'] == 'action':
                trajectory_info += f'> {item["value"]}'
            elif item['label'] == 'observation':
                trajectory_info += item['value']
            # NOT CURRENTLY SUPPORTED
            elif item['label'] == 'human_edit':
                trajectory_info += f'[human edit]: {item["value"]}'
            if i != len(self._history) - 1:
                trajectory_info += '\n'
        trajectory_info += "\n\"\"\"\n"
        return trajectory_info

    def check_is_exhausted(self) -> bool:
        return self._is_exhausted
        # return False

    def reset(self) -> None:
        self._history = []

    def __str__(self) -> str:
        s: str = self._cur_query + '\n'
        for i, item in enumerate(self._history):
            if item['label'] == 'action':
                s += f'> {item["value"]}'
            elif item['label'] == 'observation':
                s += item['value']
            # NOT CURRENTLY SUPPORTED
            elif item['label'] == 'human_edit':
                s += f'[human edit]: {item["value"]}'
            if i != len(self._history) - 1:
                s += '\n'
        return s

def _get_base_query(base_query: str, start_info: str, memory: List[str], failed_trajectory,
                    success_trajectory: List[str] = [], use_comparison=False) -> str:
    query = base_query
    # query += "Remember, You can only pick up one object at a time. If you want to pick up another object at the same time, you must first put down the object in your hand. "
    # query += "You should remember the locations you've been looking for before, don't search repeatedly for places you've already looked for.\n"
    # if len(success_trajectory) > 1:
    #     query += f"\nHere are the history experience of two successfully completed tasks. Learn something from it to guide your current task completion."
    #     query += f"\nSuccessful experience of task 1:"
    #     query += f"\n{success_trajectory[-1]}"
    #     query += f"\nSuccessful experience of task 2:"
    #     query += f"\n{success_trajectory[-2]}"
    # elif len(success_trajectory) == 1:
    #     query += f"\nHere is the history experience of one successfully completed task. Learn something from it to guide your current task completion."
    #     query += f"\nSuccessful experience of the task:"
    #     query += f"\n{success_trajectory[-1]}"

    # if use_comparison:
    #     # add memory if it exists
    #     query += f"\nHere is the current task:\n{start_info}"
    #     if len(memory) > 0:
    #         query += '\n\nYour reflexion for current task are in following three quotes:'
    #         query += "\n\"\"\"\n"
    #         for i, m in enumerate(memory):
    #             query += f"\nTrial {i}:\n"
    #             query += f'Successful experiences learned from other tasks:\n{m["successful experience"].strip()}'
    #             query += f'\nFailure reasons for current task:\n{m["failure reason"].strip()}'
    #             query += f'\nSuggestions for completing this task:\n{m["suggestions"].strip()}'
    #         query += "\n\"\"\"\n"
    #         # if len(failed_trajectory) > 0:
    #         #     query += "The trajectory sequence of the last time you attempted to solve this task but failed are in following <>. This is a failed trial and you should learn something from this failure to improve the possibility of solving this task."
    #         #     query += "\n<\n"
    #         #     query += failed_trajectory
    #         #     query += "\n>\n"
    # else:
    query += f"\nHere is the current task:\n{start_info}"
    if len(memory) > 0:
        query += '\n\nYour reflexion for the task below:'
        for i, m in enumerate(memory):
            query += f'\nTrial {i}:\n{m.strip()}'
    return query

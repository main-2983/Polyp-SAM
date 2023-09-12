from ..base import PromptBaseDataset


class PromptPolypDataset(PromptBaseDataset):
    def __init__(self, *args, task_number=0, **kwargs):
        super(PromptPolypDataset, self).__init__(*args, task_number=task_number, **kwargs)

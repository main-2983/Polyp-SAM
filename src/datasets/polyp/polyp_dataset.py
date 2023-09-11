from ..base import PromptBaseDataset


class PromptPolypDataset(PromptBaseDataset):
    def __init__(self, *args, **kwargs):
        super(PromptPolypDataset, self).__init__(*args, **kwargs)

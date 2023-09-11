from ..base import PromptBaseDataset


class PromptLesionDataset(PromptBaseDataset):
    def __init__(self, *args, **kwargs):
        super(PromptLesionDataset, self).__init__(*args, **kwargs)
from torch.utils.data import Dataset


class MainDataset(Dataset):
    def __init__(self, config):
        super(MainDataset, self).__init__()
        self.srcDir = config.srcDir

    def __getitem__(self, index):
        return False

    def __len__(self):
        return len(self.allItems)

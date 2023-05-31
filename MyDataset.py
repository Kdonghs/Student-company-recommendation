from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, df, eval_mode):
        self.df = df
        self.eval_mode = eval_mode
        if self.eval_mode:
            self.df = self.df.values.tolist()
        else:
            self.df = self.df.values.tolist()
    def __getitem__(self, index):
        if self.eval_mode:
            self.x = self.df[index]
            return self.x
        else:
            self.x = self.df[index]
            return self.x
    def __len__(self):
        return len(self.df)
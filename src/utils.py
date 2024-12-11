import torch

class ContractNLIDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {}

        for key, value in self.embeddings.items():
            element_at_idx = value[idx]
            tensor_at_idx = torch.tensor(element_at_idx)
            item[key] = tensor_at_idx

        label_at_idx = self.labels[idx]
        label_tensor = torch.tensor(int(label_at_idx))
        item["labels"] = label_tensor

        return item
    
class ContractNLIDatasetTest(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
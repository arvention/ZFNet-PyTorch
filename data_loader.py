import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class Hdf5Dataset(Dataset):

    def __init__(self, data_path, x_key, y_key):
        """
        Initialize dataset
        """

        # get data
        data_file = h5py.File(data_path, 'r')
        self.x = data_file[x_key]
        self.y = data_file[y_key]
        self.N = self.x.shape[0]

        # transform data
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def __len__(self):
        """
        Number of data in the dataset
        """
        return self.N

    def __getitem__(self, index):
        """
        Return item from dataset
        """
        image = self.x[index]
        label = self.y[index]
        return self.transform(image), torch.from_numpy(label).long()


def get_loader(data_path, x_key, y_key, batch_size, mode='train'):
    """
    Get dataset loader
    """
    dataset = Hdf5Dataset(data_path, x_key, y_key)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)

    return data_loader

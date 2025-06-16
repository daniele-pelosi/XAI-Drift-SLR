import h5py
from torch.utils.data import Dataset
import torch


class H5Dataset(Dataset):
    # dataloader = DataLoader(H5Dataset("test.h5"), batch_size=32, num_workers=0, shuffle=True)
    def __init__(self, h5_path, x_name, y_name):
        """
        Constructor.

        Parameters
        ----------
        h5_path:
            Location of hdf5 file.
        x_name: str
            Key of the input data in the hdf5 file.
        y_name: str
            Key of the target data in the hdf5 file.
        """
        self.h5_file = h5py.File(h5_path, "r")
        self.x_name = x_name
        self.y_name = y_name

    def __getitem__(self, index):
        """
        Get item by index.

        Parameters
        ----------
        index: int

        Returns
        -------
        tuple of numpy arrays. First element is the input data, second element is the target data.
        """
        return (
            self.h5_file[self.x_name][index],
            self.h5_file[self.y_name][index],
        )

    def __len__(self):
        return self.h5_file[self.y_name].size

    def get_feature_dim(self):
        return self.h5_file[self.x_name].shape[-1]

    def get_num_timesteps(self):
        return self.h5_file[self.x_name].shape[1]

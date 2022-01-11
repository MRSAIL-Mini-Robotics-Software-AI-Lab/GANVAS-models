import os
import yaml
import torch

import numpy as np

from torch.utils.data import Dataset, DataLoader

class ImgDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, n_bits:int=None):
        if n_bits is not None:
            max_val = 2**n_bits - 1
            images = np.round(images*max_val).astype(np.float)

        self.images = torch.from_numpy(np.array(images,dtype=np.float)).type(torch.FloatTensor)
        self.labels = torch.from_numpy(np.array(labels,dtype=int)).type(torch.LongTensor)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx:int):
        return self.images[idx], self.labels[idx]

def iter_wrapper(dataloader: DataLoader):
    '''
    Used on the train_loader to allow for looping on iterations instead of looping over the train_loader
    (count in iterations and once the train_loader is finished it starts over)

    Parameters
    ----------
    dataloader: torch.utils.data.DataLoader
    '''
    while True:
        for data in dataloader:
            yield data

def download_dataset(download_id:str):
    import shutil
    import requests
    from glob import glob

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : download_id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : download_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    dirname = os.path.dirname(__file__)
    destination = os.path.join(dirname,"data.zip")
    temp_dir = os.path.join(dirname,"temp")
    data_folder = os.path.join(dirname,"data")

    print("Downloading Dataset...")
    save_response_content(response, destination)

    print("Download Dataset Successfully, Unzipping")
    shutil.unpack_archive(destination, temp_dir)

    # Create data folder
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    
    print("Moving files to correct folders")
    # Move files into data folder
    files = glob(os.path.join(temp_dir,"*","*.pkl"))
    for f in files:
        name = os.path.basename(f)
        new_path = os.path.join(data_folder, name)
        if os.path.isfile(new_path):
            print(new_path, " already exists, skipping moving it")
        else:
            os.rename(f, new_path)
    
    print("Cleaning up")
    for f in glob(os.path.join(temp_dir,"*","*.")):
        os.remove(f)
    os.remove(destination)

    for f in glob(os.path.join(temp_dir,"*")):
        os.rmdir(f)
    os.rmdir(temp_dir)


'''
Steps to add your own dataset:
- Create a dictionary with a structure similar to the datasets in the data folder and save it as a pkl file
    - The dictionary should have 4 items (train, test, train_labels, test_labels), the labels can be all zeros if you wish (one zero for each image)
    train, test should be numpy arrays with images of the training data
    Ex: dictionary['train'] should have shape (num_train_images, num_channels, height, width)
    dictionary['test'] should have shape (num_test_images, num_channels, height, width)
    dictionary['train_labels'] should have shape (num_train_images, 1) filled with all zeros
    dictionary['test_labels'] should have shape (num_test_images, 1) filled with all zeros
- Name the pkl file "your_dataset_name.pkl" and add it to the data folder
- Add your_dataset_name to the assert in the first line of the function below
- Add a config for your_dataset_name in the datasets_configs.yaml (you only need the img_size parameter)
'''
def load_dataset(dataset_name:str, batch_size:int=16, n_bits:int=1):
    '''
    Downloads the dataset if needed and returns some data about the dataset (img_size, num_of_iters)

    Parameters
    ----------
    dataset_name: str
        Name of the dataset to load
    batch_size: int
        Batch size to use for train, valid, and test datasets
    n_bits: int
        Number of bits to reduce the data to (Ex: n_bits=1 -> all pixels will either be 0 or 1)
    
    Returns
    -------
    img_size: (int, int, int)
        Shape of the images in the dataset (number_of_channels, height, width)
    train_loader: Generator

    val_loader: DataLoader
        DataLoader with validation data (currently the same as the testing data)
    test_loader: DataLoader
        DataLoader with validation data (currently the same as the testing data)
    '''
    assert dataset_name in ['mnist', 'shapes','colored_mnist','colored_shapes']  # Still have to add celeb_32
    
    dirname = os.path.dirname(__file__)
    config_path = os.path.join(dirname, "datasets_configs.yaml")
    with open(config_path, 'r') as stream:
        datasets_configs = yaml.safe_load(stream)

    img_size = datasets_configs[dataset_name]['img_size']
    
    # Load images and labels from a pkl file (not suitable for large datasets)
    dataset_path = "{}.pkl".format(dataset_name)
    dataset_path = os.path.join(dirname, "data", dataset_path)
    try:
        data = np.load(dataset_path, allow_pickle=True)
    except:
        # Download the dataset
        download_dataset(datasets_configs[dataset_name]['gdrive_id'])
        data = np.load(dataset_path, allow_pickle=True)
    
    train_imgs, val_imgs, test_imgs = data['train'], data['test'], data['test']
    train_labels, val_labels, test_labels = data['train_labels'], data['test_labels'], data['test_labels']

    # Create Datasets
    train_dataset = ImgDataset(train_imgs, train_labels, n_bits=n_bits)
    val_dataset = ImgDataset(val_imgs, val_labels, n_bits=n_bits)
    test_dataset = ImgDataset(test_imgs, test_labels, n_bits=n_bits)

    # Create DataLoaders
    train_loader = iter_wrapper(DataLoader(train_dataset, batch_size))
    val_loader = DataLoader(val_dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size)

    return img_size, (train_loader, val_loader, test_loader)
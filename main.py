import argparse
from utils import *
from models import *
from datasets import *

'''
To Create a new model:
- Create YourModel.py with class YourModel
- Add line (from .YourModel import YourModel) to __init__.py in models folder
- Add folder in configs folder named YourModel
- Add a yaml file for each dataset named accordingly (see example yaml file to see required parameters)
- You can train the model by running this file [python main.py --configs ./configs/YourModel/dataset_name.yaml]
'''

def main():
    parser = argparse.ArgumentParser(description="Autoregressive trainer")
    parser.add_argument("--configs", default='./configs/MADE/shapes.yaml', type=str, help="Path to the config file")
    file_path = parser.parse_args().configs
    configs = ConfigParser(file_path)
    print("Loaded Configs")

    print("Loading Dataset")
    img_size, loaded_datasets = load_dataset(configs.dataset, configs.batch_size, configs.n_bits)
    configs.img_size = img_size
    print("Dataset Loaded")

    print("Starting Logger")
    logger = Logger(configs)
    print("Logger Ready")

    print("Creating Model")
    model_command = configs.model+"(configs, loaded_datasets, logger)"
    model = eval(model_command)
    print("Model created successfully")

    if configs.use_cuda:
        print("Using GPU")
        model = model.cuda()
    
    print("Starting Training")
    model.full_train()

if __name__ == '__main__':
    main()
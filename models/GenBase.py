import torch

import numpy as np
import torch.nn as nn

from tqdm import tqdm

class GenBase(nn.Module):
    '''
    Base Generative model that implements training, sampling, and logging
    
    Parameters
    ----------
    configs: ConfigParser
        Contains configurations for the model, dataset, and logging
    datasets: tuple of (Generator, DataLoader, DataLoader)
        Generator is the train_loader wrapper in an iter_wrapper (see datasets.py)
        The other 2 DataLoader are the val_loader and test_loader
    logger: Logger
        Used for logging using tensorboard, neptune, tqdm, and saving logs to npy files
    '''

    def __init__(self, configs, datasets, logger):
        super().__init__()
        self.configs = configs
        self.datasets = datasets
        self.logger = logger

    def loss(self, inputs):
        '''
        Calculates the loss (usually negative log likelihood) given a set of inputs

        Parameters
        ----------
        inputs: torch.FloatTensor, shape = (batch_size, num_channels, height, width)

        Returns
        -------
        torch float representing total or average loss
        '''
        raise NotImplementedError

    def sample(self, num_to_gen: int):
        '''
        Samples new images

        Parameters
        ----------
        num_to_gen: int
            Number of images to generate
        
        Returns
        -------
        np.ndarray, shape = (num_to_gen, num_channels, height, width)
        All values must be floats between 0 and 1
        '''
        raise NotImplementedError

    def full_train(self):
        configs = self.configs
        device = next(self.parameters()).device
        train_loader, val_loader, test_loader = self.datasets
        optimizer = torch.optim.Adam(self.parameters(),lr=float(configs.lr))

        val_loss = self.full_eval(val_loader)

        self.logger.log('metrics/val/loss',val_loss)

        for idx_iter in tqdm(range(1, configs.num_iters + 1), position=0, leave=True, disable=not configs.log_tqdm):
            inps, targets = next(train_loader)
            inps, targets = inps.to(device), targets.to(device)
            
            optimizer.zero_grad()
            loss = self.loss(inps)
            loss.backward()
            if configs.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), configs.clip_grad)
            optimizer.step()

            # Log training loss
            self.logger.log('metrics/train/loss',loss.item())

            # Save model if path is specified and some iterations have passed
            if configs.save_model_path is not None:
                if idx_iter%configs.save_every == 0:
                    torch.save(self, configs.save_model_path)
            
            # Eval model if some iterations have passed
            if idx_iter%configs.eval_every == 0:
                val_loss = self.full_eval(val_loader)
                self.logger.log('metrics/val/loss',val_loss)
            
            # Generate samples if some iterations have passed
            if idx_iter%configs.generate_every == 0:
                self.eval()
                generated_samples = self.sample(configs.eval_num_gen)
                self.train()  # Ensure .sample doesn't change the model into eval mode

                self.logger.log('metrics/train/samples',generated_samples)

        self.eval()
        final_test_loss = self.full_eval(test_loader)

        final_generated_samples = self.sample(configs.final_num_gen)
        self.eval()  # Ensure .sample doesn't change the model into train mode

        self.logger.log('metrics/train/samples',final_generated_samples)
        self.logger.log('metrics/test/loss',final_test_loss)

        self.logger.close()
        
        # Save final model
        if configs.save_model_path is not None:
            torch.save(self, configs.save_model_path)

    def full_eval(self, data_loader:torch.utils.data.DataLoader)->float:
        '''
        Evaluate the model on a test or validation set

        Parameters
        ----------
        self: nn.Module with a function loss that takes imgs as input
        data_loader: torch DataLoader

        Returns
        -------
        float: average test loss over all test examples
        '''
        device = next(self.parameters()).device
        test_loss = 0
        num_test = 0

        was_training = self.training

        self.eval()
        with torch.no_grad():
            for imgs, lbls in data_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)

                loss = self.loss(imgs)
                test_loss += loss.item()*imgs.shape[0]
                num_test += imgs.shape[0]
        
        if was_training:
            self.train()
        return test_loss/num_test

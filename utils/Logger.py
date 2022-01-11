import numpy as np
import neptune.new as neptune

class Logger:
    '''
    Logger class used to log data and generated samples using neptune, tensorboard, or saving the experiment to numpy arrays
    
    '''
    def __init__(self, configs):
        self.var_names = []
        self.data = {}
        self.configs = configs

        # Prepare Logging Tools (Neptune)
        if configs.log_neptune:
            self.run = neptune.init(configs.neptune_proj_name, api_token=configs.api_token, source_files=["**/*.py","**/*.yaml","**/*.yml"])
            self.run['parameters'] = configs.params
            self.run['experiment_name'] = configs.experiment_name
            self.run['tags'] = configs.neptune_tags
            self.samples_count = 0

        # Prepare Logging Tools (Tensorboard)
        if configs.log_tensorboard:
            raise NotImplementedError

    def log(self, name:str, val, type=None):
        '''
        Logs data (training/testing loss or generated samples)
        If logging samples, shape of val needs to be (num_generated, num_channels, height, width)
        '''
        if name not in self.var_names:
            self.var_names.append(name)
            self.data[name] = []
        
        self.data[name].append(val)

        if self.configs.log_neptune:
            if 'samples' in name:
                samples_name = name+'/'+str(self.samples_count)
                for sample in val:
                    to_log_img = neptune.types.File.as_image(sample.transpose(1,2,0))
                    self.run[samples_name].log(to_log_img)
                self.samples_count += 1
            else:
                self.run[name].log(val)
        
        if self.configs.log_tensorboard:
            raise NotImplementedError
    
    def close(self):
        if self.configs.log_file:
            to_save = self.data
            to_save['params'] = self.configs
            np.save(self.configs.experiment_name, to_save, allow_pickle=True)
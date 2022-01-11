import yaml

class ConfigParser:
    def __init__(self, config_path:str):
        with open(config_path,"r") as f:
            params = yaml.safe_load(f)
        self.params = params

        # Check parameters
        self.check_dataset_params()
        self.check_logging_params()
        
        for key in params.keys():
            setattr(self,key,params[key])
    
    def check_dataset_params(self):
        self.params['n_bits'] = int(self.params['n_bits'])
        pass

    def check_logging_params(self):
        pass


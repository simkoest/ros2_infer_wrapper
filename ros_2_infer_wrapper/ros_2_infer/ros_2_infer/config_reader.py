import yaml

class ConfigReader():
    def __init__(self,filename):
        self.filename = filename
        self.configs = []

    def read_conf(self):
        with open(self.filename,"r") as yamlfile:
            self.configs = yaml.load(yamlfile, Loader=yaml.FullLoader)
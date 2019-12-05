class Architecture: 
    def __init__(self, arch):
        self.__dict__.update(arch)


class Configuration:
    def __init__(self, config, arch):
        self.__dict__.update(config)
        arch['input_features'] = config['node_labels'] + 1
        self.architecture = Architecture(arch)




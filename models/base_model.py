import torch
import torch.nn as nn
import layers.modules as modules
import layers.suffix as suffix

###



##### CODE FROM github.com/hadarser/ProvablyPowerfulGraphNetworks_torch #####

class BaseModel(nn.Module):
    def __init__(self, config):
        """
        Build the model computation graph, until scores/values are returned at the end
        """
        super().__init__()

        self.config = config
        block_features = config.architecture.block_features  # List of number of features in each regular block
        original_features_num = config.node_labels + 1  # Number of features of the input

        # First part - sequential mlp blocks
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            mlp_block = modules.RegularBlock(config, last_layer_features, next_layer_features)
            self.reg_blocks.append(mlp_block)
            last_layer_features = next_layer_features

##### END OF CODE FROM github.com/hadarser/ProvablyPowerfulGraphNetworks_torch #####

        self.suffix = suffix.AverageSuffix()

    def forward(self, x):
        #here x.shape = (bs, n_vertices, n_vertices, n_features)
        x = x.permute(0, 3, 1, 2)
        #expects x.shape = (bs, n_features, n_vertices, n_vertices)
        for block in self.reg_blocks:
            x = block(x)
        x = self.suffix(x)
        # here x.shape = (bs, n_features, n_vertices)
        x = x.permute(0, 2, 1)
        # here x.shape = (bs,n_vertices, n_features)
        return x

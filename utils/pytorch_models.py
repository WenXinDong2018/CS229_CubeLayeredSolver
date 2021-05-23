import torch.nn as nn
import torch.nn.functional as F
import torch

##two idea
#1. add three separate fully connected layers to the last shared layer
#2. reduce the size of the network, spend more parameters on layer 3
class ResnetModelBaseline(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool):
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm

        # first two hidden layers
        if one_hot_depth > 0:
            self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, h1_dim)
        else:
            self.fc1 = nn.Linear(self.state_dim, h1_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            if self.batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        # output
        self.fc_out = nn.Linear(resnet_dim, out_dim)

    def forward(self, states_nnet):
        x = states_nnet

        # preprocess input
        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()

        # first two hidden layers
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)

        x = F.relu(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)

        x = F.relu(x)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)

            x = F.relu(x + res_inp)

        # output
        x = self.fc_out(x)
        return x



class ResnetModel(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool):
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm

        # first two hidden layers
        if one_hot_depth > 0:
            self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, h1_dim)
        else:
            self.fc1 = nn.Linear(self.state_dim, h1_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks-1): #one less block
            if self.batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        # output

        #wenxin: add one more layer
        # self.fc_out1 = nn.Linear(resnet_dim, resnet_dim)
        # self.fc_out2 = nn.Linear(resnet_dim, resnet_dim)
        # self.fc_out3 = nn.Linear(resnet_dim, resnet_dim)
        self.fc_out1 = nn.ModuleList([nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim), nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim)])
        self.fc_out2 = nn.ModuleList([nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim), nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim)])
        self.fc_out3 = nn.ModuleList([nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim), nn.Linear(resnet_dim, resnet_dim), nn.BatchNorm1d(resnet_dim)])

        self.out_l1 = nn.Linear(resnet_dim, 1)
        self.out_l2 = nn.Linear(resnet_dim, 1)
        self.out_l3 = nn.Linear(resnet_dim, 1)

    def forward(self, states_nnet):
        x = states_nnet

        # preprocess input
        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()

        # first two hidden layers
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)

        x = F.relu(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)

        x = F.relu(x)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)

            x = F.relu(x + res_inp)

        # output
        # l1 = self.fc_out1(x)
        # l2 = self.fc_out2(x)
        # l3 = self.fc_out3(x)

        l1 = self.fc_out1[0](x)
        l1 = self.fc_out1[1](l1)
        l1 = F.relu(l1)
        l1 = self.fc_out1[2](l1)
        l1 = self.fc_out1[3](l1)

        l2 = self.fc_out2[0](x)
        l2 = self.fc_out2[1](l2)
        l2 = F.relu(l2)
        l2 = self.fc_out2[2](l2)
        l2 = self.fc_out2[3](l2)

        l3 = self.fc_out3[0](x)
        l3 = self.fc_out3[1](l3)
        l3 = F.relu(l3)
        l3 = self.fc_out3[2](l3)
        l3 = self.fc_out3[3](l3)


        l1 = self.out_l1(l1)
        l2 = self.out_l2(l2)
        l3 = self.out_l3(l3)
        final = torch.stack((l1, l2, l3)).squeeze().T
        return final

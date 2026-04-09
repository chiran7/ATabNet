import torch
from torch.nn import Linear, BatchNorm1d, ReLU, LeakyReLU, BatchNorm2d, Sigmoid, Tanh
import numpy as np
from pytorch_tabnet import sparsemax
import torch.nn as nn
from torch.nn import Conv1d, MaxPool1d, Conv2d, Dropout
from torch import flatten
import torch.nn.functional as F


def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)


import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Linear, ReLU

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa)
        x = x * sa

        return x

class SqueezeExcite1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x.unsqueeze(-1)).squeeze(-1)
        y = self.fc(y).unsqueeze(-1)
        return x * y.squeeze(-1)

# ---- Include your modules here or import from your project ----
# from your_module import FeatTransformer, AttentiveTransformer, CBAM, SqueezeExcite1D

# ---- Multi-Branch Feature Transformer ----
class MultiBranchFeatTransformer(nn.Module):
    def __init__(self, shared_layers, input_dim, output_dim, n_branches=3,
                 n_glu_independent=2, virtual_batch_size=128, momentum=0.02):
        super().__init__()
        self.branches = nn.ModuleList([
            FeatTransformer(
                shared_layers,
                input_dim,
                output_dim,
                n_glu_independent=n_glu_independent,
                virtual_batch_size=virtual_batch_size,
                momentum=momentum
            ) for _ in range(n_branches)
        ])

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        x = torch.mean(torch.stack(branch_outputs), dim=0)
        return x

# ---- Main Encoder ----
class TabNetEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        n_branches=3  # number of branches for multi-branch transformer
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.n_branches = n_branches

        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=0.01)

        # Build shared layers
        if self.n_shared > 0:
            shared_feat_transform = nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(
                        nn.Linear(input_dim, 2 * (n_d + n_a), bias=False)
                    )
                else:
                    shared_feat_transform.append(
                        nn.Linear(n_d + n_a, 2 * (n_d + n_a), bias=False)
                    )
        else:
            shared_feat_transform = None

        self.initial_splitter = FeatTransformer(
            shared_feat_transform,
            input_dim,
            n_d + n_a,
            n_glu_independent=n_independent,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )

        self.feat_transformers = nn.ModuleList()
        self.att_transformers = nn.ModuleList()

        for step in range(n_steps):
            self.feat_transformers.append(
                MultiBranchFeatTransformer(
                    shared_layers=shared_feat_transform,
                    input_dim=input_dim,
                    output_dim=n_d + n_a,
                    n_branches=n_branches,
                    n_glu_independent=n_independent,
                    virtual_batch_size=virtual_batch_size,
                    momentum=momentum
                )
            )
            self.att_transformers.append(
                AttentiveTransformer(
                    n_a,
                    input_dim,
                    virtual_batch_size=virtual_batch_size,
                    momentum=momentum,
                    mask_type=mask_type,
                )
            )

        # Conv layers + CBAM
        self.conv1 = nn.Conv2d(10, 16, 3, stride=3)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.cbam = CBAM(channels=64)

        # SE block
        self.se_block = SqueezeExcite1D(input_dim)

    def forward(self, x, prior=None, return_masks=False):
        x = self.initial_bn(x)

        if prior is None:
            prior = torch.ones_like(x)

        M_loss = 0
        att = self.initial_splitter(x)[:, self.n_d:]
        steps_output = []

        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            ###all_masks.append(M)
            M_loss += torch.mean(torch.sum(M * torch.log(M + self.epsilon), dim=1))

            # Conv + CBAM path on M
            M_reshaped = M.reshape(M.shape[0], 10, 25, 25)
            out_conv = F.relu(self.bn1(self.conv1(M_reshaped)))
            out_conv = F.relu(self.bn2(self.conv2(out_conv)))
            out_conv = F.relu(self.bn3(self.conv3(out_conv)))
            out_conv = self.cbam(out_conv)

            flattinp = torch.flatten(out_conv, start_dim=1)
            diminp = flattinp.shape[1]
            olinear = nn.Linear(diminp, M.shape[1]).to(x.device)
            M = olinear(flattinp)

            M = (self.gamma - M) * prior
            masked_x = M * x

            # Optional SE
            # masked_x = self.se_block(masked_x)

            out = self.feat_transformers[step](masked_x)
            d = F.relu(out[:, :self.n_d])
            steps_output.append(d)
            att = out[:, self.n_d:]

        M_loss /= self.n_steps

        #if return_masks:
        #    return steps_output, M_loss, all_masks
        #else:
        return steps_output, M_loss

    def forward_masks(self, x):
        x = self.initial_bn(x)

        prior = torch.ones_like(x)
        M_explain = torch.zeros_like(x)
        att = self.initial_splitter(x)[:, self.n_d:]
        masks = {}

        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            masks[step] = M
            prior = (self.gamma - M) * prior
            masked_x = M * x
            out = self.feat_transformers[step](masked_x)
            d = F.relu(out[:, :self.n_d])
            step_importance = torch.sum(d, dim=1)
            M_explain += M * step_importance.unsqueeze(1)
            att = out[:, self.n_d:]

        return M_explain, masks


class TabNetDecoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        n_d=8,
        n_steps=3,
        n_independent=2,
        n_shared=2,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        """
        Defines main part of the TabNet network without the embedding layers.

        Parameters
        ----------
        input_dim : int
            Number of features
        output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        """
        super(TabNetDecoder, self).__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_steps = n_steps
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size

        self.feat_transformers = torch.nn.ModuleList()
        self.reconstruction_layers = torch.nn.ModuleList()

        if self.n_shared > 0:
            shared_feat_transform = torch.nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(Linear(n_d, 2 * n_d, bias=False))
                else:
                    shared_feat_transform.append(Linear(n_d, 2 * n_d, bias=False))

        else:
            shared_feat_transform = None

        for step in range(n_steps):
            transformer = FeatTransformer(
                n_d,
                n_d,
                shared_feat_transform,
                n_glu_independent=self.n_independent,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
            self.feat_transformers.append(transformer)
            reconstruction_layer = Linear(n_d, self.input_dim, bias=False)
            initialize_non_glu(reconstruction_layer, n_d, self.input_dim)
            self.reconstruction_layers.append(reconstruction_layer)

    def forward(self, steps_output):
        res = 0
        for step_nb, step_output in enumerate(steps_output):
            x = self.feat_transformers[step_nb](step_output)
            x = self.reconstruction_layers[step_nb](step_output)
            res = torch.add(res, x)
            tanhdec = nn.Tanh()            
            resnew = tanhdec(res)   
        return resnew            
        #return res
	
		

class TabNetPretraining(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        pretraining_ratio=0.2,
        n_d=1024,
        n_a=1024,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
    ):
        super(TabNetPretraining, self).__init__()

        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.pretraining_ratio = pretraining_ratio

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim

        self.masker = RandomObfuscator(self.pretraining_ratio)
        self.encoder = TabNetEncoder(
            input_dim=self.post_embed_dim,
            output_dim=self.post_embed_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
        )
        self.decoder = TabNetDecoder(
            self.post_embed_dim,
            n_d=n_d,
            n_steps=n_steps,
            n_independent=n_independent,
            n_shared=n_shared,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )
        self.discriminator = TabNetDiscriminator(
            input_dim=self.post_embed_dim,
            output_dim=self.post_embed_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
        )
		
		
		
    def forward(self, x):
        """
        Returns: res, embedded_x, obf_vars
            res : output of reconstruction
            embedded_x : embedded input
            obf_vars : which variable where obfuscated
        """
        embedded_x = self.embedder(x)
        if self.training:
            masked_x, obf_vars = self.masker(embedded_x)
            # set prior of encoder with obf_mask
            prior = 1 - obf_vars
            steps_out, _ = self.encoder(masked_x, prior=prior)
            res = self.decoder(steps_out)
            return res, embedded_x, obf_vars
        else:
            steps_out, _ = self.encoder(embedded_x)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')			
            #steps_outs = torch.stack(steps_out).to(device)	
            #steps_outs = steps_out.cpu().detach().numpy()	      		
            steps_outs = torch.sum(torch.stack(steps_out, dim=0), dim=0)	
            print('shape of encoder output at encoder definition',steps_outs.shape)
			
            res = self.decoder(steps_out)
            #resout = torch.stack(res).to(device)				
            print('shape of decoder output at encoder definition',res.shape)			
            return res, embedded_x, torch.ones(embedded_x.shape).to(x.device)

    def forward_masks(self, x):
        embedded_x = self.embedder(x)
        return self.encoder.forward_masks(embedded_x)


class TabNetNoEmbeddings(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
    ):
        """
        Defines main part of the TabNet network without the embedding layers.

        Parameters
        ----------
        input_dim : int
            Number of features
        output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(TabNetNoEmbeddings, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = BatchNorm1d(self.input_dim, momentum=0.01)

        self.encoder = TabNetEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
        )

        if self.is_multi_task:
            self.multi_task_mappings = torch.nn.ModuleList()
            for task_dim in output_dim:
                task_mapping = Linear(n_d, task_dim, bias=False)
                initialize_non_glu(task_mapping, n_d, task_dim)
                self.multi_task_mappings.append(task_mapping)
        else:
            self.final_mapping = Linear(n_d, output_dim, bias=False)
            initialize_non_glu(self.final_mapping, n_d, output_dim)

    def forward(self, x):
        res = 0
		
		
        steps_output, M_loss = self.encoder(x)
        #if return_masks:
        #    steps_output, M_loss, masks = self.encoder(x, return_masks=True)
        #else:
        #    steps_output, M_loss = self.encoder(x)

        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)

        if self.is_multi_task:
            # Result will be in list format
            out = []
            for task_mapping in self.multi_task_mappings:
                out.append(task_mapping(res))
        else:
            out = self.final_mapping(res)
        return out, M_loss
        #if return_masks:
        #    return out, M_loss, masks
        #else:
        #    return out, M_loss

    def forward_masks(self, x):
        return self.encoder.forward_masks(x)


class TabNet(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
    ):
        """
        Defines TabNet network

        Parameters
        ----------
        input_dim : int
            Initial number of features
        output_dim : int
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        cat_idxs : list of int
            Index of each categorical column in the dataset
        cat_dims : list of int
            Number of categories in each categorical column
        cat_emb_dim : int or list of int
            Size of the embedding of categorical features
            if int, all categorical features will have same embedding size
            if list of int, every corresponding feature will have specific size
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(TabNet, self).__init__()
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim
        self.tabnet = TabNetNoEmbeddings(
            self.post_embed_dim,
            output_dim,
            n_d,
            n_a,
            n_steps,
            gamma,
            n_independent,
            n_shared,
            epsilon,
            virtual_batch_size,
            momentum,
            mask_type,
        )

    def forward(self, x):
        x = self.embedder(x)
        return self.tabnet(x)

    def forward_masks(self, x):
        x = self.embedder(x)
        return self.tabnet.forward_masks(x)


class AttentiveTransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
    ):
        """
        Initialize an attention transformer.

        Parameters
        ----------
        input_dim : int
            Input size
        output_dim : int
            Output_size
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(AttentiveTransformer, self).__init__()
        self.fc = Linear(input_dim, output_dim, bias=False)
        self.conv1d= Conv1d(input_dim, output_dim,3,stride=2)
        self.maxpool1d=MaxPool1d(3,stride=2)
        #self.flatten=flatten(x)
        initialize_non_glu(self.fc, input_dim, output_dim)
        self.bn = GBN(
            output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
        )

        if mask_type == "sparsemax":
            # Sparsemax
            self.selector = sparsemax.Sparsemax(dim=-1)
        elif mask_type == "entmax":
            # Entmax
            self.selector = sparsemax.Entmax15(dim=-1)
        else:
            raise NotImplementedError(
                "Please choose either sparsemax" + "or entmax as masktype"
            )

    def forward(self, priors, processed_feat):

        x = self.fc(processed_feat)
        x = self.bn(x)				
        x = torch.mul(x, priors)
        # print('shape before selector in attentive',x.shape)			
        x = self.selector(x)
        # print('shape after selector in attentive',x.shape)		
						
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F

# Depthwise Convolution + Pointwise Projection
class DepthwiseConv1D(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding=kernel_size//2)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.unsqueeze(2)  # (B, dim, 1) ? simulate 1D conv
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = out.squeeze(2)
        return self.norm(out)

# Gated GLU Fusion
class GatedFuse(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        return gate * x1 + (1 - gate) * x2

# Final FeatTransformer
class FeatTransformer(nn.Module):
    def __init__(
        self,
        shared_layers,
        input_dim,
        output_dim,
        n_glu_independent,
        virtual_batch_size=128,
        momentum=0.02,
        use_cnn=True,
    ):
        super().__init__()

        is_first = True
        self.use_cnn = use_cnn

        if shared_layers is None:
            self.shared = nn.Identity()
        else:
            self.shared = GLU_Block(
                input_dim=input_dim,
                output_dim=output_dim,
                first=is_first,
                shared_layers=shared_layers,
                n_glu=len(shared_layers),
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
            )
            is_first = False
            #self.shared = GRN_Block(
            #    input_dim=input_dim,
            #    output_dim=output_dim,
            #    n_blocks=len(shared_layers),
            #    dropout=0.1,
            #)
            #is_first = False

        if n_glu_independent > 0:
            spec_input_dim = input_dim if is_first else output_dim
            #self.specifics = GLU_Block(
            #    input_dim=spec_input_dim,
            #    output_dim=output_dim,
            #    first=is_first,
            #    n_glu=n_glu_independent,
            #    virtual_batch_size=virtual_batch_size,
            #    momentum=momentum,
            #)
            self.specifics = GRN_Block(
                input_dim=spec_input_dim,
                output_dim=output_dim,
                n_blocks=n_glu_independent,
                dropout=0.1
            )
        else:
            self.specifics = nn.Identity()

        self.fuse = GatedFuse(dim=output_dim)

        if self.use_cnn:
            self.cnn_refine = DepthwiseConv1D(dim=output_dim)

        self.resnorm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x_shared = self.shared(x)
        x_specific = self.specifics(x_shared)
        x_fused = self.fuse(x_shared, x_specific)

        if self.use_cnn:
            x_fused = self.cnn_refine(x_fused)

        x = self.resnorm(x_fused + x_shared)  # Residual for stability
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: [batch, channels, seq_len]
        w = x.mean(dim=2)  # Global average pooling over seq_len
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).unsqueeze(2)
        return x * w

class CNNRefinementBlock(nn.Module):
    def __init__(self, channels, kernel_sizes=[3,5], dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            self.convs.append(
                nn.Conv1d(channels, channels, kernel_size=k, padding=padding, groups=channels)
            )
            self.bns.append(nn.BatchNorm1d(channels))
        self.se = SEBlock(channels)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_shared = self.shared(x)
        x_specific = self.specifics(x_shared)

        # LayerNorm before fusion
        x_shared_norm = self.layernorm(x_shared)
        x_specific_norm = self.layernorm(x_specific)

        # Dynamic gated fusion
        x_fused = self.fuse(x_shared_norm, x_specific_norm)

        # CNN expects [batch, channels, seq_len], so add channel dim = 1
        x_cnn_in = x_fused.unsqueeze(1)  # shape: [batch, 1, features]

        # CNN refinement
        x_cnn_out = self.cnn_refine(x_cnn_in)  # [batch, 1, features]

        # Remove channel dim and add residual connection
        x_cnn_out = x_cnn_out.squeeze(1)  # [batch, features]
        x_out = self.layernorm(x_cnn_out + x_fused)

        return x_out

    #def forward(self, x):
        ### x shape: [batch, channels, seq_len]
        #out = x
        #for conv, bn in zip(self.convs, self.bns):
        #    residual = out
        #    out = conv(out)
        #    out = bn(out)
        #    out = F.relu(out)
        #    out = self.dropout(out)
        #    out = out + residual  # residual connection
        #out = self.se(out)
        ## Permute for LayerNorm: [batch, seq_len, channels]
        #out = out.permute(0, 2, 1)
        #out = self.layernorm(out)
        ### Back to [batch, channels, seq_len]
        #out = out.permute(0, 2, 1)
        #return out

class DynamicGatedFusion(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x_shared, x_specific):
        # x_shared, x_specific: [batch, dim]
        concat = torch.cat([x_shared, x_specific], dim=1)
        gate = self.mlp(concat)  # [batch, dim]
        return gate * x_shared + (1 - gate) * x_specific

class GatedFuse(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x_shared, x_specific):
        x_cat = torch.cat([x_shared, x_specific], dim=-1)  # shape: [batch, 1024]
        gate = self.gate(x_cat)  # shape: [batch, 512]
        return gate * x_specific + (1 - gate) * x_shared


class MLPFuse(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x_shared, x_specific):
        x_cat = torch.cat([x_shared, x_specific], dim=-1)
        return self.fc(x_cat)

class AttentionFuse(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x_shared, x_specific):
        q = self.query(x_shared)
        k = self.key(x_specific)
        v = self.value(x_specific)
        attn = torch.softmax(torch.bmm(q.unsqueeze(1), k.unsqueeze(2)) / (q.size(-1)**0.5), dim=-1)
        fused = attn.squeeze(-1) * v
        return fused + x_shared  # residual connection

class FuseLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # start with equal weight

    def forward(self, x_shared, x_specific):
        return self.alpha * x_shared + (1 - self.alpha) * x_specific


class AGLU(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(AGLU, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.context_fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

        # Optional: add self-attention-like context gate
        self.attn = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate = self.attn(x)  # Learn gate from input context
        out = self.fc(x)
        gated_out = out * gate
        return self.norm(self.dropout(gated_out))

class AGLU_Block(nn.Module):
    def __init__(self, input_dim, output_dim, n_blocks=2, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(AGLU(input_dim, output_dim, dropout))
        for _ in range(1, n_blocks):
            self.blocks.append(AGLU(output_dim, output_dim, dropout))

    def forward(self, x):
        scale = torch.sqrt(torch.tensor(0.5).to(x.device))
        for block in self.blocks:
            res = x
            x = block(x)
            x = (x + res) * scale
        return x

class GRN(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.gate = nn.GLU(dim=-1)  # Gating with GLU
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        residual = self.skip(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.gate(torch.cat([x, x], dim=-1))  # Simple GLU gate on same features
        x = x + residual
        return self.norm(x)

class GRN_Block(nn.Module):
    def __init__(self, input_dim, output_dim, n_blocks=2, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(GRN(input_dim, output_dim, output_dim, dropout=dropout))
        for _ in range(1, n_blocks):
            self.blocks.append(GRN(output_dim, output_dim, output_dim, dropout=dropout))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class GLU_Block(torch.nn.Module):
    """
    Independent GLU block, specific to each step
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        n_glu=2,
        first=False,
        shared_layers=None,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super(GLU_Block, self).__init__()
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = torch.nn.ModuleList()
        #self.glu_layers = GRN_Block(input_dim, output_dim, n_blocks=n_glu, dropout=0.1)


        params = {"virtual_batch_size": virtual_batch_size, "momentum": momentum}

        fc = shared_layers[0] if shared_layers else None
        self.glu_layers.append(GLU_Layer(input_dim, output_dim, fc=fc, **params))
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLU_Layer(output_dim, output_dim, fc=fc, **params))

    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
        if self.first:  # the first layer of the block has no scale multiplication
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * scale
        return x


class GLU_Layer(torch.nn.Module):
    def __init__(
        self, input_dim, output_dim, fc=None, virtual_batch_size=128, momentum=0.02
    ):
        super(GLU_Layer, self).__init__()

        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = Linear(input_dim, 2 * output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2 * output_dim)

        #self.bn = GBN(
        #    2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
        #)
        self.bn = nn.BatchNorm1d(2 * output_dim, momentum=momentum)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, : self.output_dim], torch.sigmoid(x[:, self.output_dim :]))
        #return out
        return self.dropout(out)


class EmbeddingGenerator(torch.nn.Module):
    """
    Classical embeddings generator
    """

    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dim):
        """This is an embedding module for an entire set of features

        Parameters
        ----------
        input_dim : int
            Number of features coming as input (number of columns)
        cat_dims : list of int
            Number of modalities for each categorial features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs
        cat_emb_dim : int or list of int
            Embedding dimension for each categorical features
            If int, the same embedding dimension will be used for all categorical features
        """
        super(EmbeddingGenerator, self).__init__()
        if cat_dims == [] or cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            return

        self.skip_embedding = False
        if isinstance(cat_emb_dim, int):
            self.cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
        else:
            self.cat_emb_dims = cat_emb_dim

        # check that all embeddings are provided
        if len(self.cat_emb_dims) != len(cat_dims):
            msg = f"""cat_emb_dim and cat_dims must be lists of same length, got {len(self.cat_emb_dims)}
                      and {len(cat_dims)}"""
            raise ValueError(msg)
        self.post_embed_dim = int(
            input_dim + np.sum(self.cat_emb_dims) - len(self.cat_emb_dims)
        )

        self.embeddings = torch.nn.ModuleList()

        # Sort dims by cat_idx
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        self.cat_emb_dims = [self.cat_emb_dims[i] for i in sorted_idxs]

        for cat_dim, emb_dim in zip(cat_dims, self.cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))

        # record continuous indices
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

    def forward(self, x):
        """
        Apply embeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x

        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous:
                cols.append(x[:, feat_init_idx].float().view(-1, 1))
            else:
                cols.append(
                    self.embeddings[cat_feat_counter](x[:, feat_init_idx].long())
                )
                cat_feat_counter += 1
        # concat
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings


class RandomObfuscator(torch.nn.Module):
    """
    Create and applies obfuscation masks
    """

    def __init__(self, pretraining_ratio):
        """
        This create random obfuscation for self suppervised pretraining
        Parameters
        ----------
        pretraining_ratio : float
            Ratio of feature to randomly discard for reconstruction
        """
        super(RandomObfuscator, self).__init__()
        self.pretraining_ratio = pretraining_ratio

    def forward(self, x):
        """
        Generate random obfuscation mask.

        Returns
        -------
        masked input and obfuscated variables.
        """
        obfuscated_vars = torch.bernoulli(
            self.pretraining_ratio * torch.ones(x.shape)
        ).to(x.device)
        masked_input = torch.mul(1 - obfuscated_vars, x)
        return masked_input, obfuscated_vars

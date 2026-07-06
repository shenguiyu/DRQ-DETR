"""   
reference  
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
   
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""     
 
import torch
import torch.nn as nn
import torch.nn.functional as F     
import os, re    
from .common import FrozenBatchNorm2d   
from ..core import register     
from ..extre_module.custom_nn.attention.ema import EMA   
from ..extre_module.custom_nn.attention.simam import SimAM 
from ..extre_module.custom_nn.attention.CDFA import ContrastDrivenFeatureAggregation
from ..extre_module.custom_nn.attention.MSDCA import MSDCA  
from ..misc.dist_utils import Multiprocess_sync, is_dist_available_and_initialized
import logging    
  
# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_    
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_
  
__all__ = ['HGNetv2']
   

class LearnableAffineBlock(nn.Module):    
    """   
    可学习的仿射变换模块 (Learnable Affine Block)
    
    该模块对输入 `x` 进行仿射变换：
        y = scale * x + bias  
    其中 `scale` 和 `bias` 是可训练参数。
     
    适用于需要简单线性变换的场景，例如：
    - 归一化调整
    - 特征平移缩放   
    - 作为更复杂模型的一部分  
    """     
    def __init__(  
            self,
            scale_value=1.0,  
            bias_value=0.0    
    ):   
        super().__init__()  
        
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)  

    def forward(self, x):    
        """    
        前向传播：执行仿射变换

        参数:  
        x (Tensor) - 输入张量    

        返回:   
        Tensor - 变换后的输出张量
        """
        return self.scale * x + self.bias     


class ConvBNAct(nn.Module):
    def __init__(
            self,  
            in_chs,    
            out_chs,     
            kernel_size, 
            stride=1,
            groups=1, 
            padding='',
            use_act=True,
            use_lab=False
    ):  
        super().__init__()  
        self.use_act = use_act   
        self.use_lab = use_lab    
        if padding == 'same':
            self.conv = nn.Sequential(   
                
             
                nn.ZeroPad2d([0, 1, 0, 1]), 
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size,
                    stride,
                    groups=groups,
                    bias=False 
                )
            )  
        else:
            self.conv = nn.Conv2d(  
                in_chs,   
                out_chs,
                kernel_size,
                stride,  
                padding=(kernel_size - 1) // 2, 
                groups=groups,
                bias=False
            )
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()     
        else:   
            self.lab = nn.Identity()  
 
    def forward(self, x):
        x = self.conv(x)    
        x = self.bn(x)   
        x = self.act(x) 
        x = self.lab(x)  
        return x


class LightConvBNAct(nn.Module):  
    def __init__(
            self,
            in_chs,  
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False,    
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,     
            out_chs,  
            kernel_size=1,   
            use_act=False,
            use_lab=use_lab,
        )
        self.conv2 = ConvBNAct(
            out_chs, 
            out_chs,  
            kernel_size=kernel_size,  
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,     
        )
  
    def forward(self, x):    
        x = self.conv1(x) 
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    """     
    HGNetv2 的 Stem Block，作为网络的输入处理模块。   
    
    主要作用：     
    1. 进行初步特征提取（stem1）。  
    2. 通过两条不同路径（stem2a 和 stem2b）处理特征，并进行特征融合。   
    3. 采用池化（pool）和卷积（stem3、stem4）进一步提取特征。
    4. 适用于深度神经网络的前几层，以减少计算量并提高模型稳定性。   
 
    Args:     
        in_chs (int): 输入通道数。    
        mid_chs (int): 中间层通道数。
        out_chs (int): 输出通道数。
        use_lab (bool): 是否使用 Lab 颜色空间的特殊处理（默认为 False)。
    """
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False): 
        super().__init__() 
        
        self.stem1 = ConvBNAct(
            in_chs, 
            mid_chs,     
            kernel_size=3,   
            stride=2,     
            use_lab=use_lab,
        )   
  
        
        
        self.stem2a = ConvBNAct(
            mid_chs,
            mid_chs // 2,     
            kernel_size=2,
            stride=1, 
            use_lab=use_lab,  
        ) 
        
        self.stem2b = ConvBNAct( 
            mid_chs // 2,
            mid_chs,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )   
     
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True) 

        
        
        self.stem3 = ConvBNAct(
            mid_chs * 2,  
            mid_chs,   
            kernel_size=3,     
            stride=2,
            use_lab=use_lab,
        )    

        
        self.stem4 = ConvBNAct(
            mid_chs,
            out_chs,
            kernel_size=1,
            stride=1, 
            use_lab=use_lab,     
        )

    def forward(self, x):
        """
        前向传播过程：   
        1. 通过 stem1 进行特征提取。 
        2. 通过 stem2a 和 stem2b 进行一个额外的特征变换路径（右分支）。
        3. 通过最大池化（左分支）。
        4. 拼接两个分支的特征（cat）。
        5. 通过 stem3 和 stem4 进行进一步特征提取。     
        
        Args:     
            x (torch.Tensor): 输入特征图 (batch_size, in_chs, H, W)
     
        Returns:    
            torch.Tensor: 处理后的特征图 (batch_size, out_chs, H // 4, W // 4)
        """
        
        x = self.stem1(x)
        
        
        x = F.pad(x, (0, 1, 0, 1))  
        x2 = self.stem2a(x)    
        x2 = F.pad(x2, (0, 1, 0, 1))  
        x2 = self.stem2b(x2)  
        
        
        x1 = self.pool(x)

        
        x = torch.cat([x1, x2], dim=1)  
    
        
        x = self.stem3(x)  
        x = self.stem4(x)  
        
        return x


class EseModule(nn.Module):
    def __init__(self, chs):    
        super().__init__()
        self.conv = nn.Conv2d( 
            chs,
            chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()     

    def forward(self, x):     
        identity = x  
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)   
        x = self.sigmoid(x)  
        return torch.mul(identity, x)


class HG_Block(nn.Module):
    def __init__(
            self,
            in_chs,        
            mid_chs,       
            out_chs,       
            layer_num,     
            kernel_size=3, 
            residual=False, 
            light_block=False, 
            use_lab=False, 
            agg='ese',     
            drop_path=0.,  
    ):
        super().__init__()
        self.residual = residual     
   
        
        self.layers = nn.ModuleList()
        for i in range(layer_num):   
            if light_block:     
                
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,   
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab, 
                    )
                )    
            else:  
                
                self.layers.append(  
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,  
                        mid_chs,     
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                    ) 
                )   

        
        total_chs = in_chs + layer_num * mid_chs  
  
        
        if agg == 'se':  
            aggregation_squeeze_conv = ConvBNAct(
                total_chs, out_chs // 2, kernel_size=1, stride=1, use_lab=use_lab  
            )   
            aggregation_excitation_conv = ConvBNAct(  
                out_chs // 2, out_chs, kernel_size=1, stride=1, use_lab=use_lab   
            )    
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,   
                aggregation_excitation_conv,
            )
        elif agg == 'ese':  
            aggregation_conv = ConvBNAct(   
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab
            )
            att = EseModule(out_chs)  
            self.aggregation = nn.Sequential(  
                aggregation_conv,
                att,
            )
        elif agg == 'ema': 
            aggregation_conv = ConvBNAct(   
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab 
            )     
            att = EMA(out_chs)    
            self.aggregation = nn.Sequential(   
                aggregation_conv,    
                att,
            )   
        elif agg == 'simam': 
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab
            )
            att = SimAM()
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att, 
            )
        elif agg == 'CDFA':
            aggregation_conv = ConvBNAct( 
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab     
            ) 
            att = ContrastDrivenFeatureAggregation(out_chs)  
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )
        elif agg == 'msdca': 
            aggregation_conv = ConvBNAct( 
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab     
            ) 
            att = MSDCA(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )    
        else:
            raise Exception(f"param agg{agg} Illegal.")     
   
        
        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()   

    def forward(self, x):
        identity = x  
        output = [x]  
        for layer in self.layers:
            x = layer(x)
            output.append(x)  
        
        # x1 = layers_1(x)
        # x2 = layers_2(x1) 
        # x3 = layers_3(x2)  
        # output = [x, x1, x2, x3]  
    
        
        x = torch.cat(output, dim=1)

        
        x = self.aggregation(x)
     
        
        if self.residual:   
            x = self.drop_path(x) + identity
        
        return x


class HG_Stage(nn.Module): 
    """    
    HG_Stage 模块 (Hourglass Stage)
     
    该模块由多个 HG_Block 组成，并可选进行下采样（downsample）。
     
    主要功能： 
    1. 可选地对输入特征进行下采样，以减少空间尺寸，提高感受野。
    2. 通过多个 HG_Block 进行特征提取，每个块逐层累积特征。   
    3. 适用于分层特征提取任务，如目标检测、分割和回归任务。

    参数:
    - in_chs (int): 输入通道数
    - mid_chs (int): 中间通道数（隐层维度） 
    - out_chs (int): 输出通道数 
    - block_num (int): 该 stage 内包含的 HG_Block 数量 
    - layer_num (int): 每个 HG_Block 内的层数
    - downsample (bool): 是否在进入 Stage 之前进行下采样（默认 True）
    - light_block (bool): 是否使用轻量级卷积块（影响 HG_Block 内部结构） 
    - kernel_size (int): 卷积核大小（默认为 3）
    - use_lab (bool): 是否使用可学习的仿射变换模块
    - agg (str): 特征聚合方式，可选 'se' 或 'ese' 或 'ema' 或 'simam'
    - drop_path (float or list): DropPath 比例（用于防止过拟合）

    """ 
 
    def __init__(     
            self,  
            in_chs,        
            mid_chs,       
            out_chs,       
            block_num,     
            layer_num,     
            downsample=True, 
            light_block=False, 
            kernel_size=3, 
            use_lab=False, 
            agg='se',      
            drop_path=0.,  
    ):
        super().__init__()
        self.downsample = downsample 
 
        
        if downsample:
            self.downsample = ConvBNAct( 
                in_chs,   
                in_chs,   
                kernel_size=3,
                stride=2,  
                groups=in_chs,  
                use_act=False,  
                use_lab=use_lab, 
            )
        else:
            self.downsample = nn.Identity()  

        
        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block( 
                    in_chs if i == 0 else out_chs,  
                    mid_chs,   
                    out_chs,   
                    layer_num, 
                    residual=False if i == 0 else True, 
                    kernel_size=kernel_size, 
                    light_block=light_block, 
                    use_lab=use_lab, 
                    agg=agg, 
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path, 
                ) 
            )  
        
        
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        """     
        前向传播过程：     
        1. 先进行可选的下采样   
        2. 通过多个 HG_Block 进行特征提取
        """   
        x = self.downsample(x)  
        x = self.blocks(x)  
        return x
   
 
     
@register()
class HGNetv2(nn.Module):
    """
    HGNetV2  
    Args:     
        stem_channels: list. Number of channels for the stem block. 
        stage_type: str. The stage configuration of HGNet. such as the number of channels, stride, etc.  
        use_lab: boolean. Whether to use LearnableAffineBlock in network.
        lr_mult_list: list. Control the learning rate of different stages.
    Returns: 
        model: nn.Layer. Specific HGNetV2 model depends on args.
    """
 
    arch_configs = {
        'B0': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num     
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],  
            },     
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        }, 
        'B1': {     
            'stem_channels': [3, 24, 32],   
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num   
                "stage1": [32, 32, 64, 1, False, False, 3, 3],     
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],    
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'
        },   
        'B2': {
            'stem_channels': [3, 24, 32],
            'stage_config': {   
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],    
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],   
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'
        },   
        'B3': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num    
                "stage1": [32, 32, 128, 1, False, False, 3, 5],  
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5], 
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],    
            },    
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'
        },   
        'B4': {   
            'stem_channels': [3, 32, 48],
            'stage_config': { 
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],    
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'
        },   
        'B5': {
            'stem_channels': [3, 32, 64],
            'stage_config': {  
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6], 
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth' 
        },    
        'B6': {
            'stem_channels': [3, 48, 96],  
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num  
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6], 
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],    
            },    
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth'
        },   
    }     

    def __init__(self,  
                 name,    
                 use_lab=False,  
                 return_idx=[1, 2, 3], 
                 freeze_stem_only=True, 
                 freeze_at=0, 
                 freeze_norm=True,
                 pretrained=True,
                 agg='se',    
                 local_model_dir='weight/hgnetv2/'):
        super().__init__()  
        self.use_lab = use_lab  
        self.return_idx = return_idx
     
        stem_channels = self.arch_configs[name]['stem_channels']  
        stage_config = self.arch_configs[name]['stage_config']
        download_url = self.arch_configs[name]['url']

        self._out_strides = [4, 8, 16, 32]  
        self._out_channels = [stage_config[k][2] for k in stage_config]
     
        # stem 
        self.stem = StemBlock(
                in_chs=stem_channels[0],   
                mid_chs=stem_channels[1],
                out_chs=stem_channels[2],   
                use_lab=use_lab)     
 
        # stages
        self.stages = nn.ModuleList() 
        for i, k in enumerate(stage_config):     
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[
                k]
            self.stages.append(    
                HG_Stage(  
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num, 
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size, 
                    use_lab,  
                    agg))    
     
        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):    
                    self._freeze_parameters(self.stages[i])
   
        if freeze_norm:
            self._freeze_norm(self)
   
        if pretrained:     
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"     
            try: 
                model_path = local_model_dir + 'PPHGNetV2_' + name + '_stage1.pth'
                if os.path.exists(model_path): 
                    state = torch.load(model_path, map_location='cpu')
                    print(f"Loaded stage1 {name} HGNetV2 from local file.")   
                else:
                    # If the file doesn't exist locally, download from the URL
                    if is_dist_available_and_initialized() and torch.distributed.get_rank() == 0:
                        print(GREEN + "If the pretrained HGNetV2 can't be downloaded automatically. Please check your network connection." + RESET)
                        print(GREEN + "Please check your network connection. Or download the model manually from " + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                        state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu', model_dir=local_model_dir)   
                        Multiprocess_sync()
                    else:  
                        Multiprocess_sync()
                        state = torch.load(local_model_dir)     
   
                    print(f"Loaded stage1 {name} HGNetV2 from URL.")

                # need_keep_key_prefix_list = ['stem.*', 'stages.0.*']
                # need_pop_key_list = [] 
                # for key in state.keys():
                #     need_pop = True
                #     for keep in need_keep_key_prefix_list: 
                #         if re.match(keep, key):
                #             need_pop = False     
                #             break
                #     if need_pop: 
                #         need_pop_key_list.append(key)    
                # for key in need_pop_key_list:     
                #     state.pop(key)  
  
                print(RED + f'Loading Pretrained State Dict Key Names:{state.keys()}' + RESET)
                    
                self.load_state_dict(state, strict=False)    
 
            except (Exception, KeyboardInterrupt) as e:     
                if (is_dist_available_and_initialized() and torch.distributed.get_rank() == 0) or (not is_dist_available_and_initialized()):
                    print(f"{str(e)}") 
                    logging.error(RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET)     
                    logging.error(GREEN + "Please check your network connection. Or download the model manually from "\
                                + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                exit()

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):  
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():   
                _child = self._freeze_norm(child)
                if _child is not child:   
                    setattr(m, name, _child)
        return m 
  
    def _freeze_parameters(self, m: nn.Module):   
        for p in m.parameters():
            p.requires_grad = False
     
    def forward(self, x):    
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):    
            x = stage(x) 
            if idx in self.return_idx:  
                outs.append(x)    
        return outs  

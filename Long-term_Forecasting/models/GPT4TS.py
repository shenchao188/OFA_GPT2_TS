import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
import os


class GPT4TS(nn.Module):

    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1  # 前面是patchTST的一些操作

        if configs.is_gpt:
            if configs.pretrain:
                # self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
                # 这里我们直接修改了模型加载的方式，原本的方式直接从网上下载GPT2模型非常的耗时，因此我们想直接下载到本地，然后后续的实验只需要加载该模型即可

                # <editor-fold desc="折叠后要显示的内容">
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                # 首次运行时下载并保存
                if not os.path.exists('./pretrained_model/local_gpt2'):
                    model = GPT2Model.from_pretrained('gpt2')
                    model.save_pretrained('./pretrained_model/local_gpt2')
                # 后续运行直接加载本地文件
                self.gpt2 = GPT2Model.from_pretrained('./pretrained_model/local_gpt2',
                                                      local_files_only=True,
                                                      output_attentions=True,
                                                      output_hidden_states=True)

                # </editor-fold>



            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]  # h 是一个ModuleList，包含了所有的transformer层，默认的GPT-2模型有12
            # 层transformer块，相当于取了前6层来做预测
            print("gpt2 = {}".format(self.gpt2))

        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)  # d_model=768,即GPT2的嵌入维度/隐藏层维度
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)  # 不采用循环迭代预测，而是直接输出对应的预测长度

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True  # layer normalization层的参数以及word position embedding层的参数的参数需要训练
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()

        self.cnt = 0

    def forward(self, x, itr):
        B, L, M = x.shape  # L是输入时间段的长度；B是batchsize；M应该是变量数量

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev  # 又做了一次标准化？

        x = rearrange(x, 'b l m -> b m l')  # 调整特征与时间点两个维度

        x = self.padding_patch_layer(x)  # 时间序列填充
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # 这就是patch操作
        x = rearrange(x, 'b m n p -> (b m) n p')

        outputs = self.in_layer(x)#线性层做维度上的对齐
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs

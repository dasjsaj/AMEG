import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from test import RNN1
from test import G2ANet
from test import BootstrappedRNN
import torch

# 设置matplotlib的后端
matplotlib.use('TkAgg')

# 加载模型
test = BootstrappedRNN(42,64,16,3,9)
test1 = BootstrappedRNN(48,64,16,4,9)
test2 = G2ANet(42, 64,32, 9)
test3 = G2ANet(48, 64,32, 9)
# test2 = RNN1(102, 64, 14)
# test3 = RNN1(108, 64, 14)
test.load_state_dict(torch.load(r"./model/maven/3m/rnn_net_params.pkl", map_location='cuda:0'))
test1.load_state_dict(torch.load(r"./model/maven/4m_vs_3m/rnn_net_params.pkl", map_location='cuda:0'))
test2.load_state_dict(torch.load(r"./model/reinforce+g2anet/3m/rnn_params.pkl", map_location='cuda:0'))
test3.load_state_dict(torch.load(r"./model/reinforce+g2anet/4m_vs_3m/rnn_params.pkl", map_location='cuda:0'))

# 创建图形和子图
fig, axs = plt.subplots(1, 4, figsize=(20, 5),sharey=True, gridspec_kw={'wspace': 0.000005})  # 设置figsize确保一行可以放下四个子图

labels = ['MAVEN/3m', 'MAVEN/4m_vs_3m', 'Reinforce+G2ANET/3m', 'Reinforce+G2ANET/4m_vs_3m']
models = [test.fc.weight, test1.fc.weight, test2.encoding.weight, test3.encoding.weight]

for i, ax in enumerate(axs):
    # 创建一个热图
    im = ax.imshow(models[i].detach().numpy(), cmap='viridis')

    # 设置标题
    ax.set_title(labels[i])

    # 隐藏坐标轴刻度
    ax.axis('off')

# 添加颜色条
# fig.colorbar(im, ax=axs, orientation='horizontal', pad=0.05)

# 调整布局
plt.tight_layout()
plt.savefig(r'E:\实验室\实验室\博士论文\博士论文撰写\7_第七篇\visio\仿真\SMAC\model\rein-maven-3m.jpg', dpi=1000, bbox_inches='tight',pad_inches = 0)
# 显示图形
plt.show()

# 保存图像


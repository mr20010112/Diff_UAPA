import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Beta

# 定义 Beta 分布的参数
alpha_values = [1.0, 2.5, 5.0, 10.0]
beta_values = [1.0, 2.5, 5.0, 10.0]

# 定义 x 的取值范围
x = torch.linspace(0.01, 0.99, 500, dtype=torch.float64) # 避免 0 和 1（Beta 分布在这些点上不定义）

# 创建图形
plt.figure(figsize=(10, 6))

# 遍历不同的 alpha 和 beta 值
for alpha, beta in zip(alpha_values, beta_values):
    # 创建 Beta 分布
    beta_dist = Beta(torch.tensor([alpha], dtype=torch.float64), torch.tensor([beta], dtype=torch.float64))

    # 计算 PDF 值
    pdf = torch.exp(beta_dist.log_prob(x))

    # 找到最大值及其对应的 x
    max_pdf = pdf.max()
    max_x = x[torch.argmax(pdf)]

    # 绘制曲线
    plt.plot(x.numpy(), pdf.numpy(), label=f'α={alpha}, β={beta}')

    # 标记最大值
    plt.scatter([max_x.item()], [max_pdf.item()], color='red') # 用红色点标记最值
    plt.text(max_x.item(), max_pdf.item(), f'({max_x.item():.2f}, {max_pdf.item():.2f})', fontsize=10, color='red')

# 设置图形标题和标签
plt.title('Beta Distribution PDF with Maximum Values', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig("beta_distribution.png") # 保存图片
plt.close() # 关闭图形
# plt.show()

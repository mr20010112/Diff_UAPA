import wandb
import pandas as pd

# # 登录 WandB
# wandb.login()

entity = "mr20010112-tsinghua-university"  # 确认用户名/组织名
project_name = "diffusion_policy_debug"  # 确认项目名
run_id = "349nrygw"  # 确认运行 ID

api = wandb.Api()
run = api.run(f"{entity}/{project_name}/{run_id}")  # 检查路径格式是否正确

# 导出所有记录的日志数据
history = run.history()  # 获取所有 step/epoch 的数据
history_df = pd.DataFrame(history)  # 转换为 DataFrame

# 针对某个指标计算每个 step/epoch 的均值
# index = ["test/p_1",
#          "test/p_2",
#          "test/p_3",
#          "test/p_4",
#          "test/p_5"]

index = ['test/mean_score']

for i in range(len(index)):
    mean_per_step = history_df[index[i]].mean()
    # 输出每个 step 的均值
    print(f"Mean value step/epoch{(i+1)}:", mean_per_step)

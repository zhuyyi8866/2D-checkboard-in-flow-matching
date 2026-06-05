# Flow Matching From Scratch

## 项目简介

本项目实现了一个基于 flow matching 的 2D checkerboard 生成模型。主要目标是从简单先验分布（如标准高斯）学习一个时间条件速度场，将样本沿概率路径推送到目标分布。这份代码以 `2D_checkerboard.ipynb` 为示例，展示完整的数据生成、路径采样、模型训练、ODE 采样和似然估计流程。

## 主要目录结构

- `model/path/`
  - `path.py`：`ProbPath` 抽象基类，定义概率路径接口。
  - `affine.py`：`AffinePath`，实现线性插值路径 `x_t = (1-t)x_0 + t x_1`。
  - `affine_alias.py`：`AffineProbPath` 兼容别名，供 notebook 直接使用。
  - `scheduler/`：调度器模块，包含 `CondOTScheduler` 等时间调度函数。
- `model/solver/`
  - `ode_solver.py`：`ODESolver`，用于基于训练好的速度场进行数值积分采样和似然计算。
- `utils/`
  - `model_wrapper.py`：模型封装与训练步骤工具。
  - `utils.py`：辅助张量函数。
- `2D_checkerboard.ipynb`
  - 主流程 notebook，包含 dataset、model、训练、采样与可视化。

## 实现方法

### 1. 数据生成

使用 `inf_train_gen(batch_size, device)` 作为无穷数据生成器，直接构造 2D checkerboard 样本 `x_1`。它不依赖 PyTorch Dataset/ DataLoader，而是按需生成训练样本。

### 2. 概率路径

使用 `AffineProbPath` 表示简单线性路径：
- `x_t = (1 - t) * x_0 + t * x_1`
- `dx_t = x_1 - x_0`

这样在训练时可以直接得到目标速度 `dx_t` 作为监督信号。

### 3. 调度器

`CondOTScheduler` 提供时间相关权重，用于构造更灵活的路径或在后续扩展中支持带权重的目标。这种设计使路径模块与调度模块解耦，更易替换。

### 4. 速度场模型

使用一个小型 MLP 作为速度场网络 `vf(x, t)`，它接收当前空间点和时间，并输出与输入同维度的速度向量。

### 5. 训练目标

训练损失为流匹配损失：

`L = E[||vf(x_t, t) - dx_t||^2]`

即让网络预测的速度与路径真实速度一致，从而使网络学会沿路径 transport 样本。

### 6. 采样与似然

训练完成后，通过 `ODESolver` 对速度场做数值积分，利用 `x_0` 生成新的样本并检查分布演化。Notebook 还包含基于流场积分计算 log-likelihood 的示例。

## FM_1 环境说明

本项目应当在 Conda 环境 `FM_1` 中运行。当前测试环境信息如下：

- Conda 环境：`FM_1`
- PyTorch 版本：`2.7.1+cu118`
- CUDA 可用：`True`

### 推荐依赖安装

项目根目录下已包含 `requirements.txt`，请在 `FM_1` 环境中运行：

```bash
conda activate FM_1
python -m pip install -r requirements.txt
```

`requirements.txt` 包含本项目常用依赖：

- `torch==2.7.1+cu118`
- `matplotlib`
- `numpy`
- `torchdiffeq`
- `jupyter`

### 推荐运行方式

```bash
conda activate FM_1
python -m pip install -r requirements.txt
jupyter notebook 2D_checkerboard.ipynb
```

## 运行提示

- 先打开 `2D_checkerboard.ipynb`，按顺序运行各个单元。
- 如果出现导入错误，先确认当前工作目录在项目根目录 `flow_matching_from_scretch`，并且已经激活 `FM_1`。
- 如果需要 `torchdiffeq` 才能完成 ODE 采样或似然计算，请额外安装：

```bash
pip install torchdiffeq
```

## 我的一些思考

### 1. 模型似然计算方法的思考

项目最后的似然计算不是直接通过传统生成模型的显式概率密度函数得到，而是基于速度场的逆向 ODE 积分与 log 密度跟踪。这样的方法更贴近连续正则化流（CNF）和流匹配的思想：通过解算时间反向动力学，将目标样本回退到先验，并累积变化的 log 密度。

我觉得这种方式的优点在于它可以利用同一个速度场既做采样，又做似然评估；缺点则是对数值积分误差和散度估计误差比较敏感。因此，实际使用时需要谨慎选择时间步长、求积方法与散度计算策略。

### 2. 对隐式方法的创新思考(当然这个小项目可能体现得不明显)

在传统采样阶段，显式欧拉或 midpoint 可能存在稳定性问题，特别是当速度场较大或系统接近刚性时。将隐式欧拉引入到推理过程中，是为了增强数值稳定性并容忍更大的时间步长。

隐式方法的创新点在于它本质上对下一状态进行迭代求解，可部分缓解显式方法的震荡和发散风险。虽然隐式欧拉会增加计算开销，但对于一些难以收敛的速度场，它提供了一条更稳健的采样路径，尤其适合在模型生成阶段追求更可靠的分布演化。


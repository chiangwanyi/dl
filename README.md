# dl · 深度学习练习

手写反向传播、手写优化器、手写 RK4 的小实验集合。不打包、不建包，每个 `.py` 都是一份
可以单独跑通的学习材料；两条实验线共享同一套 `.venv`。

## 环境

Python 3.15 + uv 管理虚拟环境（`.venv/`），依赖见 `pyproject.toml`。

```bash
uv sync              # 安装/更新依赖
uv add <pkg>         # 新增依赖
```

## 目录结构

```
dl/
├── 01-基础练习/                 从零手写的网络与梯度
│   ├── 01_autograd_mlp.py       两层 MLP 拟合 XOR，手动梯度下降
│   ├── 02_线性回归轨迹.py        线性回归拟合导弹轨迹（输入 t 和 t²，无隐藏层）
│   └── 021_激活函数.py           单隐层 1-16-2，tanh 激活，可换激活函数对比拟合效果
│
├── 02-抛体向量场/               「猜位置」升级成「猜力」
│   ├── 022_向量场.py             训练位置模型 t→(x,y) 与场模型 (vx,vy,k)→(ax,ay)，导出权重
│   ├── 022_向量场.html           抛体仿真页：拖 v0 / 角度 / 阻力系数，三条轨迹叠加对比
│   └── 022_model.{js,json}      由 022_向量场.py 生成，勿手改
│
├── 03-弹簧振子/                 完整一点的项目：仿真 → 造数据 → 训练 → 网页推演
    ├── README.md                实现说明、结果指标、边界条件（先读这份）
    ├── src/                     sim / dataset / model / train，纯 numpy，无 torch
    ├── web/                     4 个递进版本的仿真页 + spring_model.{js,json}
    └── .cache/                  data_cache.npz（22 MB，已 gitignore，可重新生成）
```

`01-` `02-` `03-` `04-` 是推荐阅读顺序，不是依赖关系；各目录彼此独立，改动互不影响。

### 04-手写数字识别（手写 conv / pool，不用 torch）

```
04-手写数字识别/
├── mnist_loader.py        读 idx 格式，返回 (N,28,28) uint8
├── mnist_mlp.py           全连接版（200,704 参数）
├── mnist_cnn.py           卷积版（27,538 参数），conv3x3 / maxpool2 都是手写的
├── trace_cnn.py           训练途中打权重快照 → web/trace_cnn.js（给逐层可视化页用）
├── verify_forward.js      node 校验：网页里的前向 JS == numpy 前向
├── data/                  MNIST 四个 .gz（不入库）
└── web/
    ├── cnn_forward.js      前向计算，页面和校验脚本共用这一份
    ├── index.html          MLP 手写实时预测
    ├── cnn.html            CNN 手写实时预测
    └── train.html          训练过程逐层可视化：同一张图在不同快照下的尺寸/像素变化
```

## 快速开始

```bash
# 基础练习（任意目录执行，看到图就关掉窗口）
uv run python 01-基础练习/01_autograd_mlp.py
uv run python 01-基础练习/021_激活函数.py

# 抛体向量场：训练约 3 分钟，权重写回同目录，然后直接打开 html
uv run python 02-抛体向量场/022_向量场.py
# 打开 02-抛体向量场/022_向量场.html，支持 ?v0=20&ang=40&k=0.03

# 弹簧振子
python src/train.py --reuse      # 在 03-弹簧振子/ 目录下执行
# 打开 03-弹簧振子/web/弹簧小球-轨迹预测.html

# 手写数字识别：训练 + 导出网页用的权重（约 4 分钟）
uv run python 04-手写数字识别/mnist_cnn.py
# 训练过程快照（页面要的数据就是这个生成）
uv run python 04-手写数字识别/trace_cnn.py
# 可选：校验网页前向和 numpy 是否一致
node 04-手写数字识别/verify_forward.js
# 打开 04-手写数字识别/web/train.html
```

## 约定

- **权重文件与 HTML 必须同目录**。网页用相对路径 `<script src="...">` 加载权重，
  这样 `file://` 双击打开也能跑，不依赖本地服务器。
- **权重文件是生成物**，只改 `.py` 再重跑，别手改 `.js / .json`。
- **根目录只放项目级文件**（README / pyproject / uv.lock / .gitignore），
  单次实验的文件放进对应主题目录；新开一条线就新建 `04-xxx/`。
- **全角括号不入文件名**。`022（向量场）` 这种命名在 shell 和 `sys.argv` 里都要额外转义。

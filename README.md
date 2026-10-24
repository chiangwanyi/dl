# dl

深度学习练习脚本集。不打包、不建包,每个 `.py` 是一个独立可运行的小实验。

## 环境

Python 3.15 + uv 管理虚拟环境(`.venv/`),依赖见 `pyproject.toml`。

```bash
uv sync              # 安装/更新依赖
uv add <pkg>         # 新增依赖
```

## 运行

```bash
uv run python 01_autograd_mlp.py
```

因为项目根目录在 `sys.path` 上,脚本之间可以互相 `import`,例如 `from utils import foo`。

## 脚本

| 文件 | 内容 |
|---|---|
| `01_autograd_mlp.py` | 用 autograd 手写两层 MLP 拟合 XOR,手动梯度下降 |

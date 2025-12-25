# LocalColabFold 安装指南（使用 Micromamba）

本指南将帮助你使用 micromamba 安装 LocalColabFold。

## 前提条件

1. **确保已安装基础工具**：
   ```bash
   sudo apt -y install curl git wget
   ```

2. **确保 CUDA 版本 ≥ 11.8**（推荐 12.4，如果不使用 GPU 可跳过）：
   ```bash
   nvcc --version
   ```

3. **确保 GNU 编译器版本 ≥ 12.0**（用于 `--amber` 松弛功能）

## 安装步骤

### 方式一：使用命名环境（推荐，更简单）

#### 1. 创建环境

```bash
micromamba create -n colabfold -c conda-forge -c bioconda \
    git python=3.10 openmm==8.2.0 pdbfixer \
    kalign2=2.04 hhsuite=3.3.0 mmseqs2 -y
```

#### 2. 激活环境

```bash
micromamba activate colabfold
```

### 方式二：使用路径环境（项目隔离）

如果你希望将环境安装在项目目录下，便于管理和清理：

#### 1. 设置安装目录

```bash
cd /home/davis/projects/C2C_release/localcolabfold
export COLABFOLDDIR="$(pwd)/localcolabfold"
mkdir -p "${COLABFOLDDIR}"
```

#### 2. 创建环境

```bash
micromamba create -p "$COLABFOLDDIR/colabfold-conda" -c conda-forge -c bioconda \
    git python=3.10 openmm==8.2.0 pdbfixer \
    kalign2=2.04 hhsuite=3.3.0 mmseqs2 -y
```

#### 3. 激活环境

```bash
micromamba activate "$COLABFOLDDIR/colabfold-conda"
```

---

### 后续步骤（两种方式相同）

### 安装 ColabFold 和依赖

```bash
# 安装 ColabFold
pip install --no-warn-conflicts \
    "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"

pip install "colabfold[alphafold]"

# 安装 GPU 版本的 JAX（需要 CUDA 12）
pip install --upgrade "jax[cuda12]==0.5.3"

# 安装 TensorFlow 和静默警告工具
pip install --upgrade tensorflow
pip install silence_tensorflow
```

### 修改 ColabFold 配置

```bash
# 获取环境路径
CONDA_PREFIX_PATH=$(python -c "import sys; print(sys.prefix)")

pushd "${CONDA_PREFIX_PATH}/lib/python3.10/site-packages/colabfold"

# 使用 'Agg'（可选，使用默认缓存目录）
# sed -i -e "s#appdirs.user_cache_dir(__package__ or \"colabfold\")#\"${HOME}/.cache('Agg')\nimport matplotlib.pyplot as plt#g" plot.py

# 修改默认参数目录
sed -i -e "s#appdirs.user_cache_dir(__package__ or \"colabfold\")#\"${COLABFOLDDIR}/colabfold\"#g" download.py

# 抑制 TensorFlow 警告
sed -i -e "s#from io import StringIO#from io import StringIO\nfrom silence_tensorflow import silence_tensorflow\nsilence_tensorflow()#g" batch.py

# 清除缓存
rm -rf __pycache__
popd
```

### 下载 AlphaFold2 权重

```bash
python -m colabfold.download
```

### 配置环境变量（可选）

使用命名环境时，每次使用前只需 `micromamba activate colabfold` 即可，无需配置 PATH。

## 验证安装

```bash
colabfold_batch --help
```

## 使用方法

```bash
# 基本预测
colabfold_batch input.fasta outputdir/

# 使用模板和 AMBER 松弛
colabfold_batch --templates --amber input.fasta outputdir/
```

## WSL2 用户额外配置

如果在 WSL2 中运行，添加以下环境变量：

```bash
export TF_FORCE_UNIFIED_MEMORY="1"
export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export TF_FORCE_GPU_ALLOW_GROWTH="true"
```

## 故障排除

如果 `--amber` 松弛出现错误，尝试添加：

```bash
export LD_LIBRARY_PATH="${COLABFOLDDIR}/colabfold-conda/lib:${LD_LIBRARY_PATH}"
```

## 一键安装脚本

你也可以将以下内容保存为 `install_micromamba.sh` 并运行：

```bash
#!/bin/bash -e

type wget 2>/dev/null || { echo "wget is not installed." ; exit 1 ; }

CURRENTPATH=$(pwd)
COLABFOLDDIR="${CURRENTPATH}/localcolabfold"

mkdir -p "${COLABFOLDDIR}"

# 创建环境
micromamba create -p "$COLABFOLDDIR/colabfold-conda" -c conda-forge -c bioconda \
    git python=3.10 openmm==8.2.0 pdbfixer \
    kalign2=2.04 hhsuite=3.3.0 mmseqs2 -y

# 激活环境
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$COLABFOLDDIR/colabfold-conda"

# 安装 Python 包
"$COLABFOLDDIR/colabfold-conda/bin/pip" install --no-warn-conflicts \
    "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"
"$COLABFOLDDIR/colabfold-conda/bin/pip" install "colabfold[alphafold]"
"$COLABFOLDDIR/colabfold-conda/bin/pip" install --upgrade "jax[cuda12]==0.5.3"
"$COLABFOLDDIR/colabfold-conda/bin/pip" install --upgrade tensorflow
"$COLABFOLDDIR/colabfold-conda/bin/pip" install silence_tensorflow

# 修改配置
pushd "${COLABFOLDDIR}/colabfold-conda/lib/python3.10/site-packages/colabfold"
sed -i -e "s#from matplotlib import pyplot as plt#import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt#g" plot.py
sed -i -e "s#appdirs.user_cache_dir(__package__ or \"colabfold\")#\"${COLABFOLDDIR}/colabfold\"#g" download.py
sed -i -e "s#from io import StringIO#from io import StringIO\nfrom silence_tensorflow import silence_tensorflow\nsilence_tensorflow()#g" batch.py
rm -rf __pycache__
popd

# 下载权重
"$COLABFOLDDIR/colabfold-conda/bin/python3" -m colabfold.download

echo "Installation finished!"
echo "Add to PATH: export PATH=\"${COLABFOLDDIR}/colabfold-conda/bin:\$PATH\""
```

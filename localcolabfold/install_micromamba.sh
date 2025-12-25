#!/bin/bash -e

# LocalColabFold 安装脚本（使用 Micromamba）
# 基于 INSTALL_MICROMAMBA.md 文档

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查必要工具
echo_info "检查必要工具..."

type wget 2>/dev/null || { echo_error "wget 未安装，请运行: sudo apt install wget"; exit 1; }
type curl 2>/dev/null || { echo_error "curl 未安装，请运行: sudo apt install curl"; exit 1; }
type git 2>/dev/null || { echo_error "git 未安装，请运行: sudo apt install git"; exit 1; }
type micromamba 2>/dev/null || { echo_error "micromamba 未安装，请先安装 micromamba"; exit 1; }

echo_info "所有必要工具已就绪"

# 检查 CUDA（可选）
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo_info "检测到 CUDA 版本: $CUDA_VERSION"
else
    echo_warn "未检测到 CUDA，将安装 CPU 版本"
fi

# 设置安装目录
CURRENTPATH=$(pwd)
COLABFOLDDIR="${CURRENTPATH}/localcolabfold"

echo_info "安装目录: ${COLABFOLDDIR}"

# 创建目录
mkdir -p "${COLABFOLDDIR}"

# 创建 conda 环境
echo_info "创建 micromamba 环境..."
micromamba create -p "$COLABFOLDDIR/colabfold-conda" -c conda-forge -c bioconda \
    git python=3.10 openmm==8.2.0 pdbfixer \
    kalign2=2.04 hhsuite=3.3.0 mmseqs2 -y

# 激活环境
echo_info "激活环境..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$COLABFOLDDIR/colabfold-conda"

# 设置 pip 路径
PIP="$COLABFOLDDIR/colabfold-conda/bin/pip"
PYTHON="$COLABFOLDDIR/colabfold-conda/bin/python3"

# 安装 ColabFold
echo_info "安装 ColabFold..."
"$PIP" install --no-warn-conflicts \
    "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"

"$PIP" install "colabfold[alphafold]"

# 安装 JAX（GPU 版本需要 CUDA 12）
echo_info "安装 JAX..."
if command -v nvcc &> /dev/null; then
    "$PIP" install --upgrade "jax[cuda12]==0.5.3"
else
    "$PIP" install --upgrade jax
fi

# 安装 TensorFlow 和静默警告工具
echo_info "安装 TensorFlow..."
"$PIP" install --upgrade tensorflow
"$PIP" install silence_tensorflow

# 修改 ColabFold 配置
echo_info "修改 ColabFold 配置..."
SITE_PACKAGES="${COLABFOLDDIR}/colabfold-conda/lib/python3.10/site-packages/colabfold"

pushd "${SITE_PACKAGES}" > /dev/null

# 修改 matplotlib 后端
sed -i -e "s#from matplotlib import pyplot as plt#import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt#g" plot.py

# 修改默认参数目录
sed -i -e "s#appdirs.user_cache_dir(__package__ or \"colabfold\")#\"${COLABFOLDDIR}/colabfold\"#g" download.py

# 抑制 TensorFlow 警告
sed -i -e "s#from io import StringIO#from io import StringIO\nfrom silence_tensorflow import silence_tensorflow\nsilence_tensorflow()#g" batch.py

# 清除缓存
rm -rf __pycache__

popd > /dev/null

# 下载 AlphaFold2 权重
echo_info "下载 AlphaFold2 权重（这可能需要一些时间）..."
"$PYTHON" -m colabfold.download

# 创建激活脚本
echo_info "创建激活脚本..."
cat > "${COLABFOLDDIR}/activate.sh" << EOF
#!/bin/bash
# LocalColabFold 环境激活脚本

eval "\$(micromamba shell hook --shell bash)"
micromamba activate "$COLABFOLDDIR/colabfold-conda"

# WSL2 用户可能需要的额外配置
# export TF_FORCE_UNIFIED_MEMORY="1"
# export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
# export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
# export TF_FORCE_GPU_ALLOW_GROWTH="true"

# 如果 --amber 松弛出现错误，取消下面这行的注释
# export LD_LIBRARY_PATH="${COLABFOLDDIR}/colabfold-conda/lib:\${LD_LIBRARY_PATH}"

echo "LocalColabFold 环境已激活"
echo "使用方法: colabfold_batch input.fasta outputdir/"
EOF

chmod +x "${COLABFOLDDIR}/activate.sh"

echo ""
echo_info "============================================"
echo_info "安装完成！"
echo_info "============================================"
echo ""
echo "激活环境："
echo "  source ${COLABFOLDDIR}/activate.sh"
echo ""
echo "或者手动激活："
echo "  micromamba activate ${COLABFOLDDIR}/colabfold-conda"
echo ""
echo "使用方法："
echo "  colabfold_batch input.fasta outputdir/"
echo "  colabfold_batch --templates --amber input.fasta outputdir/"
echo ""
echo "验证安装："
echo "  colabfold_batch --help"
echo ""

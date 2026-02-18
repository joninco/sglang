#!/bin/bash
# Build sglang Docker image with Blackwell (sm_120a) support for CUDA 12.9
# with transformers installed from git (main branch)
# Optimized for high-core-count build servers

set -e

IMAGE_NAME="${1:-sglang-blackwell}"
IMAGE_TAG="${2:-sm120a-tfv5}"
TRANSFORMERS_REPO="${TRANSFORMERS_REPO:-https://github.com/huggingface/transformers.git}"
TRANSFORMERS_BRANCH="${TRANSFORMERS_BRANCH:-main}"

echo "=============================================="
echo "Building sglang with Blackwell (sm_120a) support"
echo "  + transformers from git"
echo "=============================================="
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "CUDA: 12.9.1"
echo "MAX_JOBS: 128"
echo "TORCH_CUDA_ARCH_LIST: 12.0a"
echo "FLASHINFER_CUDA_ARCH_LIST: 12.0a"
echo "Transformers: ${TRANSFORMERS_REPO}@${TRANSFORMERS_BRANCH}"
echo ""

cd "$(dirname "$0")"

# Stage 1: Build the base image using the Blackwell Dockerfile
echo ">>> Stage 1: Building base Blackwell image..."
docker buildx build \
    --target framework \
    --platform linux/amd64 \
    -f docker/Dockerfile.blackwell \
    --build-arg CUDA_VERSION=12.9.1 \
    --build-arg TORCH_CUDA_ARCH_LIST=12.0a \
    --build-arg FLASHINFER_CUDA_ARCH_LIST=12.0a \
    --build-arg BUILD_SGL_KERNEL_FROM_SOURCE=1 \
    --build-arg BUILD_TYPE=all \
    --build-arg BRANCH_TYPE=local \
    --build-arg GRACE_BLACKWELL=0 \
    --build-arg INSTALL_FLASHINFER_JIT_CACHE=1 \
    --build-arg SGL_VERSION=latest \
    --build-arg BUILD_AND_DOWNLOAD_PARALLEL=128 \
    -t "${IMAGE_NAME}:${IMAGE_TAG}-base" \
    --load \
    .

# Stage 2: Install transformers from git on top
echo ""
echo ">>> Stage 2: Installing transformers from ${TRANSFORMERS_REPO}@${TRANSFORMERS_BRANCH}..."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" - <<DOCKERFILE
FROM ${IMAGE_NAME}:${IMAGE_TAG}-base
RUN pip install "git+${TRANSFORMERS_REPO}@${TRANSFORMERS_BRANCH}" \
    && python3 -c "import transformers; print(f'transformers {transformers.__version__} installed from git')"
DOCKERFILE

echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Transformers: installed from ${TRANSFORMERS_REPO}@${TRANSFORMERS_BRANCH}"
echo ""
echo "To use with docker-compose, update docker-compose.yaml:"
echo "  image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Or run directly:"
echo "  docker run --gpus all -p 8000:8000 -v /path/to/models:/models:ro ${IMAGE_NAME}:${IMAGE_TAG} \\"
echo "    python3 -m sglang.launch_server \\"
echo "    --model-path /models/GLM-4.7-FP8 \\"
echo "    --tp-size 8 \\"
echo "    --tool-call-parser glm47 \\"
echo "    --reasoning-parser glm45 \\"
echo "    --host 0.0.0.0"
echo ""
echo "To customize the transformers source:"
echo "  TRANSFORMERS_REPO=https://github.com/your-fork/transformers.git TRANSFORMERS_BRANCH=v5.0.0 ./build-blackwell-transformers-v5.sh"

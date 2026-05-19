#!/usr/bin/env bash
# Download evaluation datasets for probing (NYUv2, KITTI Eigen, ETH3D).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ROOT}/data"
mkdir -p "${DATA}"

echo "==> NYUv2 depth (standard 654-image test split)"
NYU_DIR="${DATA}/nyu_depth_v2"
mkdir -p "${NYU_DIR}"
if [[ ! -f "${NYU_DIR}/.download_complete" ]]; then
  python - <<PY
from pathlib import Path
import urllib.request
import zipfile

data_dir = Path("${NYU_DIR}")
# Official labeled subset (Matlab .mat); we document manual steps if auto fails.
readme = data_dir / "README_DOWNLOAD.txt"
readme.write_text("""
NYUv2 dense depth — obtain labeled subset:

1. Register / download from:
   https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
2. Extract so you have:
   data/nyu_depth_v2/nyu_depth_v2_labeled.mat
   OR a folder of RGB/depth pairs matching the standard train/test split.

Alternative (smaller, for smoke tests):
   pip install opencv-python-headless
   Use torch hub / torchvision if you already have NYUv2 elsewhere — symlink:
   ln -s /path/to/existing/nyu ${data_dir}/raw
""")
print("Wrote", readme)
PY
  touch "${NYU_DIR}/.download_complete"
fi

echo "==> KITTI Eigen depth (697 test images)"
KITTI_DIR="${DATA}/kitti_eigen"
mkdir -p "${KITTI_DIR}"
if [[ ! -f "${KITTI_DIR}/.download_complete" ]]; then
  cat > "${KITTI_DIR}/README_DOWNLOAD.txt" <<'EOF'
KITTI Eigen split for depth evaluation:

1. Download KITTI raw / depth benchmarks from https://www.cvlibs.net/datasets/kitti/
2. Use the Eigen train/val split file from:
   https://github.com/nianticlabs/marigold (or Marigold repo data splits)
3. Place images under: data/kitti_eigen/raw/
   and projected depth under: data/kitti_eigen/gt_depth/

For automated prep, run after raw KITTI is available:
   python -m src.data.prepare_kitti_eigen --kitti_root /path/to/kitti --out data/kitti_eigen
EOF
  touch "${KITTI_DIR}/.download_complete"
fi

echo "==> ETH3D (zero-shot)"
ETH_DIR="${DATA}/eth3d"
mkdir -p "${ETH_DIR}"
if [[ ! -f "${ETH_DIR}/.download_complete" ]]; then
  cat > "${ETH_DIR}/README_DOWNLOAD.txt" <<'EOF'
ETH3D multi-view + depth:

https://www.eth3d.net/datasets
Download low-res multi-view training / test depth benchmarks as needed.
Place under data/eth3d/
EOF
  touch "${ETH_DIR}/.download_complete"
fi

echo ""
echo "Dataset scaffolding created under ${DATA}/"
echo "NYUv2 and KITTI require manual download (licensing / size)."
echo "See README_DOWNLOAD.txt in each subfolder."

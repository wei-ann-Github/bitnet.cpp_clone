#!/usr/bin/env bash
set -euo pipefail

# Prefer CUDA 12.9 on newer Ubuntu/GCC toolchains for better compatibility.
# Override explicitly with: BITNET_CUDA_HOME=/path/to/cuda bash compile.sh
if [[ -n "${BITNET_CUDA_HOME:-}" ]]; then
	export CUDA_HOME="${BITNET_CUDA_HOME}"
else
	for c in /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12 /usr/local/cuda; do
		if [[ -x "$c/bin/nvcc" ]]; then
			export CUDA_HOME="$c"
			break
		fi
	done
fi

if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
	if command -v nvcc >/dev/null 2>&1; then
		NVCC_BIN="$(command -v nvcc)"
		CUDA_HOME="$(cd "$(dirname "$NVCC_BIN")/.." && pwd)"
		export CUDA_HOME
	else
		echo "Error: nvcc not found. Install CUDA toolkit and/or set CUDA_HOME." >&2
		echo "Hint: export CUDA_HOME=/usr/local/cuda-12.9 && export PATH=\"\$CUDA_HOME/bin:\$PATH\"" >&2
		exit 1
	fi
fi

export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

NVCC="${CUDA_HOME}/bin/nvcc"

# Prefer GCC/G++ 14 if present (CUDA 12.x often fails with GCC 15+).
CCBIN_ARGS=()
if command -v g++-14 >/dev/null 2>&1; then
	CCBIN_ARGS=(-ccbin "$(command -v g++-14)")
elif command -v g++ >/dev/null 2>&1; then
	CCBIN_ARGS=(-ccbin "$(command -v g++)")
fi

if [[ ${#CCBIN_ARGS[@]} -gt 0 ]]; then
	CXX="${CCBIN_ARGS[1]}"
	CXX_MAJOR="$(${CXX} -dumpversion 2>/dev/null | cut -d. -f1 || echo 0)"
	if [[ "${CXX_MAJOR}" -gt 14 ]]; then
		if [[ "${BITNET_ALLOW_UNSUPPORTED_COMPILER:-0}" != "1" ]]; then
			echo "Error: Host compiler is GCC ${CXX_MAJOR}, but CUDA 12.9 supports GCC <= 14." >&2
			echo "Install g++-14 and rerun, or set BITNET_ALLOW_UNSUPPORTED_COMPILER=1 to try anyway." >&2
			echo "Example: sudo apt install g++-14" >&2
			exit 1
		fi
		EXTRA_NVCC_FLAGS=(-allow-unsupported-compiler)
	else
		EXTRA_NVCC_FLAGS=()
	fi
else
	EXTRA_NVCC_FLAGS=()
fi

echo "Using CUDA_HOME=${CUDA_HOME}"
"${NVCC}" --version | head -n 4

set -x
"${NVCC}" \
	"${CCBIN_ARGS[@]}" \
	"${EXTRA_NVCC_FLAGS[@]}" \
	-U_GNU_SOURCE \
	-D_DEFAULT_SOURCE \
	-D_XOPEN_SOURCE=700 \
	-D_POSIX_C_SOURCE=200809L \
	-std=c++17 \
	-Xcudafe --diag_suppress=177 \
	--compiler-options -fPIC \
	-lineinfo \
	--shared bitnet_kernels.cu \
	-lcuda \
	-gencode=arch=compute_80,code=compute_80 \
	-o libbitnet.so
set +x

echo "Built: $(pwd)/libbitnet.so"



# Source me: environment used for every phase-1 command on the laptop (WSL2, no sudo).
# - user-local toolchain (gcc/cmake/kernel headers/libxcrypt) from micromamba env `vla-tools`, needed by
#   Triton / torch.compile and a few sdist builds (evdev); the system has no compiler.
# - a clean PATH so Windows binaries under /mnt/c (python, conda, npm) can never be picked up.
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/micromamba}"
_T="$MAMBA_ROOT_PREFIX/envs/vla-tools"
export PATH="$HOME/.local/bin:$_T/bin:/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib"
export CC="$_T/bin/gcc" CXX="$_T/bin/g++"
export CPATH="$_T/x86_64-conda-linux-gnu/sysroot/usr/include:$_T/include"
export UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# LIBERO: repo-local config (no interactive prompt, nothing written to ~/.libero); software EGL rendering
# (Mesa llvmpipe). The GPU d3d12 Mesa driver hangs creating a MuJoCo context under WSL2.
export LIBERO_CONFIG_PATH="$_REPO/.libero"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  mkdir -p "$LIBERO_CONFIG_PATH"
  _R="$_REPO/third_party/libero/libero/libero"
  printf 'benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: %s/../datasets\nassets: %s/assets\n' \
    "$_R" "$_R" "$_R" "$_R" "$_R" > "$LIBERO_CONFIG_PATH/config.yaml"
fi

#!/bin/bash
set -e

echo "=========================================="
echo "Starting Conda independent compilation and deep binding script..."
echo "=========================================="

# 1. Check Conda environment
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: No active Conda environment detected!"
    echo "Please run 'conda activate ciao-fxt' first"
    exit 1
fi
echo "Current Conda environment: $CONDA_PREFIX"

# 2. Locate the external 'contbin' source directory (Strict Parallel Verification)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPyXT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"       # EPyXT project root directory
PARALLEL_DIR="$(cd "$EPyXT_ROOT/.." && pwd)"     # Expected parallel parent directory

SOURCE_DIR="$PARALLEL_DIR/contbin"               # Strict standard: parallel directory
INTERNAL_DIR="$EPyXT_ROOT/contbin"               # Incorrect path: inside project

# Strict path validation
if [ -d "$INTERNAL_DIR" ]; then
    echo "================================================================="
    echo "Error: Detected 'contbin' was cloned INSIDE the EPyXT directory!"
    echo "This violates the required project structure layout."
    echo "================================================================="
    echo "Please move it to the parallel directory before compiling:"
    echo "  mv $INTERNAL_DIR $PARALLEL_DIR/"
    exit 1
elif [ ! -d "$SOURCE_DIR" ]; then
    echo "================================================================="
    echo "Error: Could not find 'contbin' folder in the parallel directory!"
    echo "Expected path: $SOURCE_DIR"
    echo "================================================================="
    echo "Please go to the parallel directory and clone it:"
    echo "  cd $PARALLEL_DIR"
    echo "  git clone https://github.com/jeremysanders/contbin.git"
    exit 1
fi

echo "Source path verified & locked (Parallel Standard): $SOURCE_DIR"

# 3. Start local compilation
cd "$SOURCE_DIR"
echo "Cleaning up old residuals in $SOURCE_DIR..."
make clean || true

echo "Starting compilation..."
g++ -O2 -g -Wall -std=c++11 \
-I"$CONDA_PREFIX/include" \
contbin.cc binner.cc bin.cc fitsio_simple.cc format_string.cc memimage.cc scrubber.cc terminal.cc flux_estimator.cc \
$(find parammm \( -name "*.cc" -o -name "*.cpp" \) 2>/dev/null || true) \
-L"$CONDA_PREFIX/lib" -lcfitsio \
-o contbin_new

# 4. Verify and copy into Conda
echo "Writing to Conda system toolbox..."
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
./contbin_new --help > /dev/null 2>&1 || [ $? -eq 0 ] || [ $? -eq 1 ]

cp contbin_new "$CONDA_PREFIX/bin/contbin"
chmod +x "$CONDA_PREFIX/bin/contbin"

# 5. Configure Conda environment variable hooks
echo "Configuring automatic mounting..."
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

echo 'export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
echo 'export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH' > "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh"
echo 'unset OLD_LD_LIBRARY_PATH' >> "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh"

echo "=========================================="
echo "Compilation successful and written to Conda!"
echo "Please execute: conda deactivate && conda activate ciao-fxt"
echo "=========================================="
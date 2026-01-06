# For running with C++

## Rebuild after changes to C++
1. From the repo root (where pyproject.toml lives):
```
pip install -U scikit-build-core pybind11 numpy
pip install -e .
```
This will:
* Configure CMake
* Compile `ga_member.cpp`
* Install the extension module `ga_member_cpp` in editable mode
2. Because this is a compiled extension, it must be rebuilt after editing `ga_member.cpp`.
```
# Activate the venv you are ACTUALLY using here (mine is src/.venv)
source .venv/bin/activate
rm -rf build/ dist/ *.egg-info
python -m pip install -e . -v
```
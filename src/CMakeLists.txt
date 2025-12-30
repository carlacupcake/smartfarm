cmake_minimum_required(VERSION 3.18)
project(smartfarm_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

# Build extension as a submodule inside the ga_member_cpp package
pybind11_add_module(_core smartfarm/core/cpp/ga_member.cpp)

install(TARGETS _core
  LIBRARY DESTINATION ga_member_cpp
  RUNTIME DESTINATION ga_member_cpp
  ARCHIVE DESTINATION ga_member_cpp
)

# Install the Python wrapper package (so ga_member_cpp is NOT a namespace pkg)
install(FILES smartfarm/core/python/ga_member_cpp/__init__.py
        DESTINATION ga_member_cpp)

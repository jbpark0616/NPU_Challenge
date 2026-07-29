FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 1. Update and install essential tools
RUN apt update && apt install -y \
  lsb-release software-properties-common \
  libhwloc-dev cmake ninja-build \
  hwloc libboost-all-dev \
  git git-lfs g++ build-essential \
  libtbb-dev libcapstone-dev pkg-config \
  libssl-dev wget pciutils \
  sudo \
  vim \
  zip unzip \
  python3-pip python3.10-venv libgl1


# 2. Install LLVM 17
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 17 all && \
    rm llvm.sh

# 3. Install Clang
RUN apt install -y clang-17 lldb-17 lld-17

# 4. Manually build and install CMake 3.31.8
WORKDIR /opt
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8.tar.gz && \
    tar zxvf cmake-3.31.8.tar.gz && \
    cd cmake-3.31.8 && \
    ./bootstrap && \
    make -j32 && \
    sudo make install && \
    cd .. && rm -rf cmake-3.31.8 cmake-3.31.8.tar.gz

# 5. Extract tt-metal package
WORKDIR /
RUN mkdir tt-metal
COPY tt-metal/ /tt-metal/

# 6. Final build steps
WORKDIR /tt-metal
RUN ./install_dependencies.sh --docker
RUN bash -c "source ./env_set.sh && ./build_metal.sh -p -b Release"

# 7. Install virtual environment
WORKDIR /tt-metal
RUN git config --global --add safe.directory /tt-metal
RUN ./create_venv.sh

# 8. Set working directory
WORKDIR /tt-metal


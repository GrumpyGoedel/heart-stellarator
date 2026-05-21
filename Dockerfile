FROM python:3.11-slim

# Minimal C++ toolchain to compile SIMSOPT (simsoptpp) from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_ROOT_USER_ACTION=ignore \
    MPLBACKEND=Agg \
    CMAKE_BUILD_PARALLEL_LEVEL=4

RUN pip install --no-cache-dir --timeout 300 --retries 8 \
        numpy \
        scipy \
        matplotlib \
        pillow \
        "pybind11<3" \
        scikit-build-core

# SIMSOPT: compile from source so simsoptpp targets the actual container CPU
# (the prebuilt aarch64 wheel uses extensions Colima/QEMU may not expose).
# Clone with submodules — the PyPI sdist is missing thirdparty/fmt.
RUN git clone --depth 1 --recursive \
        https://github.com/hiddenSymmetries/simsopt.git /opt/simsopt \
 && pip install --no-cache-dir --timeout 300 --retries 8 /opt/simsopt

WORKDIR /work
COPY heart_stellarator.py /work/

CMD ["python", "/work/heart_stellarator.py"]

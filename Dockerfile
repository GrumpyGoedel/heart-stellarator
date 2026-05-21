FROM python:3.11-slim

# Install build toolchain + FORTRAN legacy compilers for VMEC/SIMSOPT
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        cmake \
        git \
        wget \
        libopenmpi-dev \
        openmpi-bin \
        libnetcdf-dev \
        libnetcdff-dev \
        libhdf5-dev \
        libfftw3-dev \
        liblapack-dev \
        libblas-dev \
        libopenblas-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_ROOT_USER_ACTION=ignore \
    MPLBACKEND=Agg
RUN pip install --no-cache-dir --timeout 300 --retries 8 \
        numpy \
        scipy \
        matplotlib \
        mpi4py \
        f90nml \
        h5py \
        netcdf4 \
        "pybind11<3" \
        scikit-build-core \
        cmake

# SIMSOPT — compile from source so simsoptpp targets the actual container CPU
# (the prebuilt aarch64 wheel uses instructions Colima/QEMU may not expose).
# Clone with submodules: the PyPI sdist is missing thirdparty/fmt.
ENV CMAKE_BUILD_PARALLEL_LEVEL=4
RUN git clone --depth 1 --recursive https://github.com/hiddenSymmetries/simsopt.git /opt/simsopt \
 && pip install --no-cache-dir --timeout 300 --retries 8 /opt/simsopt
RUN pip install --no-cache-dir --timeout 300 --retries 8 --no-build-isolation \
        "git+https://github.com/hiddenSymmetries/VMEC2000.git" \
    || echo "VMEC2000 build skipped — script will fall back to pure-SIMSOPT path."

WORKDIR /work
COPY simple_plasma_coils.py /work/

CMD ["python", "/work/simple_plasma_coils.py"]

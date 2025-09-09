#!/bin/bash

set -euxo pipefail
INSTALLPREFIX="$1"
PYTBLIS_ARCH="$2"
C_COMPILER="$3"
CXX_COMPILER="$4"
PLATFORM_ID="$5"

if [[ "${PLATFORM_ID}" == "macosx_x86_64" ]]; then
  export CFLAGS="-arch x86_64"
  export CXXFLAGS="-arch x86_64"
  export GA_NCPU=4
  export THREAD_MODEL=pthreads
  export HWLOC_ENABLED=OFF
fi

if [[ "${PLATFORM_ID}" == "macosx_arm64" ]]; then
  export GA_NCPU=3
  export THREAD_MODEL=pthreads
  export HWLOC_ENABLED=OFF
fi

if [[ "${PLATFORM_ID}" == manylinux* ]]; then
  export GA_NCPU=4
  export THREAD_MODEL=pthreads
  export HWLOC_ENABLED=ON
  dnf -y install hwloc hwloc-devel
fi

cmake -S tblis -B tblisbld \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALLPREFIX}" \
  -DCMAKE_C_COMPILER="${C_COMPILER}" \
  -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
  -DBLIS_CONFIG_FAMILY="${PYTBLIS_ARCH}" \
  -DENABLE_HWLOC="${HWLOC_ENABLED}" \
  -DBLIS_THREAD_MODEL="${THREAD_MODEL}" \
  -DENABLE_THREAD_MODEL="${THREAD_MODEL}"
# cmake --build tblisbld --parallel "${GA_NCPU}" --verbose
make -C tblisbld -j "${GA_NCPU}"
cmake --install tblisbld

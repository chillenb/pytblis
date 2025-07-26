#!/bin/bash

set -euxo pipefail
INSTALLPREFIX="$1"
PYTBLIS_ARCH="$2"
C_COMPILER="$3"
CXX_COMPILER="$4"

cmake -S tblis -B tblisbld \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${INSTALLPREFIX} \
  -DCMAKE_C_COMPILER=${C_COMPILER} \
  -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
  -DBLIS_CONFIG_FAMILY=${PYTBLIS_ARCH}
cmake --build tblisbld --parallel 8
cmake --install tblisbld

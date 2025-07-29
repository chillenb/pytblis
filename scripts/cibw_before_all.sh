#!/bin/bash

set -euxo pipefail
INSTALLPREFIX="$1"
PYTBLIS_ARCH="$2"
C_COMPILER="$3"
CXX_COMPILER="$4"
PLATFORM_ID="$5"

if [[ "${PLATFORM_ID}" == "macos_x86_64" ]]; then
  export CFLAGS="-arch x86_64"
  export CXXFLAGS="-arch x86_64"
fi

cmake -S tblis -B tblisbld \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALLPREFIX}" \
  -DCMAKE_C_COMPILER="${C_COMPILER}" \
  -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
  -DBLIS_CONFIG_FAMILY="${PYTBLIS_ARCH}"
cmake --build tblisbld --parallel 8 --verbose
cmake --install tblisbld

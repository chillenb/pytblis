#!/usr/bin/env bash

# Convenient way to get a development installation

NCPUS=$(nproc)

if ! test -d local_tblis_prefix; then
  cmake -S tblis -B tblisbld \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=local_tblis_prefix \
    -DBLIS_CONFIG_FAMILY="auto" \
    -DENABLE_STATIC=OFF \
    -DENABLE_SHARED=ON
  cmake --build tblisbld --parallel "$NCPUS"
  cmake --install tblisbld
fi

CMAKE_ARGS="-DTBLIS_ROOT=local_tblis_prefix" pip install --no-cache-dir -e . -vv

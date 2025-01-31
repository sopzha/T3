#!/bin/bash
cd dp || exit
mkdir -p bin
cd bin || exit
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .

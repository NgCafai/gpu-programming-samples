mkdir -p debug
cd debug
rm -f CMakeCache.txt
cmake -DCMAKE_BUILD_TYPE=Debug ..
make clean
make
./sgemm_main 2
cd ..
mkdir -p build
cd build
rm -f CMakeCache.txt
cmake ..
make clean
make
./sgemm_main 2
cd ..
#include <iostream>
#include "kernel.h"

int main() {
    const int arraySize = 5;
    int a[arraySize] = {1, 2, 3, 4, 5};
    int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize] = {0};

    // Run the add kernel on the GPU
    runAddKernel(c, a, b, arraySize);

    // Display the results
    for (int i = 0; i < arraySize; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

    return 0;
}

#include <iostream>
#include <vector>
#include <cstdlib>

int main() {
    std::cout << "Starting memory allocation...\n";

    // Allocate a large block of memory
    int* largeArray = new int[1000000]; // Allocate 1 million integers
    for (int i = 0; i < 1000000; ++i) {
        largeArray[i] = rand();
    }

    // Simulate memory usage
    std::vector<int*> chunks;
    for (int i = 0; i < 10; ++i) {
        chunks.push_back(new int[100000]); // Allocate 100,000 integers per chunk
    }

    // Deallocate some memory
    for (int i = 0; i < 5; ++i) {
        delete[] chunks[i];
    }

    delete[] largeArray;
    std::cout << "Memory allocation complete.\n";
    return 0;
}

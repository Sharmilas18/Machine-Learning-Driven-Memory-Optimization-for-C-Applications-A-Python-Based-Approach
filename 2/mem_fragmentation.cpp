#include <iostream>
#include <cstdlib>

int main() {
    std::cout << "Testing memory fragmentation...\n";

    for (int i = 0; i < 100; ++i) {
        int* block1 = new int[1000];
        int* block2 = new int[1000];
        delete[] block1; // Create gaps in memory
        int* block3 = new int[500]; // Smaller block
        delete[] block2;
        delete[] block3;
    }

    std::cout << "Fragmentation test complete.\n";
    
    return 0;
}

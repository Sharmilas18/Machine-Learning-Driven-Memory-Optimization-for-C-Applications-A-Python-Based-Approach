#include <iostream>

void memoryLeak() {
    int* leakedMemory = new int[1000000]; // Allocate memory and never free it
}

int main() {
    std::cout << "Simulating memory leaks...\n";

    for (int i = 0; i < 100; ++i) {
        memoryLeak();
    }

    std::cout << "Memory leak simulation complete.\n";
    return 0;
}

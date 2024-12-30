#include <iostream>
#include <vector>
#include <cstdlib>

int main() {
    const int size = 1000000;
    std::vector<int> array(size);

    // Randomly populate the array
    for (int i = 0; i < size; ++i) {
        array[i] = rand();
    }

    // Access memory in a non-sequential pattern
    for (int i = size - 1; i >= 0; i -= 100) {
        array[i] += 1; // Random access
    }

    std::cout << "Cache miss simulation complete.\n";
    return 0;
}

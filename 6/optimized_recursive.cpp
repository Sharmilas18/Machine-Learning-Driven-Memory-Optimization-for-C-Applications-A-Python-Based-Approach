#include <iostream>

void recursiveFunction(int depth) {
    int largeArray[1000]; // Allocate 1000 integers on the stack
    if (depth > 0) {
        recursiveFunction(depth - 1);
    }
}

int main() {
    std::cout << "Starting recursion...\n";
    recursiveFunction(1000); // Large recursion depth
    std::cout << "Recursion complete.\n";
    return 0;
}

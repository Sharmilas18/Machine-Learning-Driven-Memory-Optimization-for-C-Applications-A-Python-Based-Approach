#include <iostream>
#include <vector>

int main() {
    std::cout << "Growing vector memory...\n";
    std::vector<int> data;

    for (int i = 0; i < 1000000; ++i) {
        data.push_back(i); // Gradually increase memory usage
    }

    std::cout << "Vector size: " << data.size() << "\n";
    return 0;
}

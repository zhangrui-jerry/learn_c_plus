#include <iostream>
#include "add.hpp"

int main(int, char**) {
    std::cout << "Hello, world!\n";
    int a = 10;
    int b = 20;
    std::cout << add(a, b) << std::endl;
}

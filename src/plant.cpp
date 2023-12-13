#include "include/plant.h"

#include <iostream>

Plant::Plant() {
    t = 0;
}

void Plant::query() {
    std::cout << "Plant has been queried." << std::endl;
}

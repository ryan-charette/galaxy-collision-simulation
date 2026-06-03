#pragma once

#include <cmath>
#include <iostream>
#include <string>

namespace fmmgalaxy::tests {

inline bool require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        return false;
    }
    return true;
}

inline bool near(double a, double b, double tolerance) {
    return std::abs(a - b) <= tolerance;
}

}  // namespace fmmgalaxy::tests

#pragma once

#include <cmath>

namespace fmmgalaxy {

struct Vec2 {
    double x{0.0};
    double y{0.0};

    constexpr Vec2() = default;
    constexpr Vec2(double x_, double y_) : x(x_), y(y_) {}

    constexpr Vec2 operator+(const Vec2& other) const { return {x + other.x, y + other.y}; }
    constexpr Vec2 operator-(const Vec2& other) const { return {x - other.x, y - other.y}; }
    constexpr Vec2 operator-() const { return {-x, -y}; }
    constexpr Vec2 operator*(double s) const { return {x * s, y * s}; }
    constexpr Vec2 operator/(double s) const { return {x / s, y / s}; }

    Vec2& operator+=(const Vec2& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    Vec2& operator-=(const Vec2& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    Vec2& operator*=(double s) {
        x *= s;
        y *= s;
        return *this;
    }

    Vec2& operator/=(double s) {
        x /= s;
        y /= s;
        return *this;
    }
};

inline constexpr Vec2 operator*(double s, const Vec2& v) {
    return v * s;
}

inline constexpr double dot(const Vec2& a, const Vec2& b) {
    return a.x * b.x + a.y * b.y;
}

inline constexpr double cross(const Vec2& a, const Vec2& b) {
    return a.x * b.y - a.y * b.x;
}

inline constexpr double norm_squared(const Vec2& v) {
    return dot(v, v);
}

inline double norm(const Vec2& v) {
    return std::sqrt(norm_squared(v));
}

}  // namespace fmmgalaxy

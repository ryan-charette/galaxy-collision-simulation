#pragma once

#include <cmath>

namespace fmmgalaxy {

/// @brief Three-component Cartesian vector used for positions, velocities, and accelerations.
///
/// The historical `Vec2` alias remains for compatibility with earlier planar code, but the
/// simulator stores and evolves all particle state in three dimensions.
struct Vec3 {
    /// Cartesian x component.
    double x{0.0};
    /// Cartesian y component.
    double y{0.0};
    /// Cartesian z component.
    double z{0.0};

    /// Construct the zero vector.
    constexpr Vec3() = default;
    /// Construct a planar vector with `z = 0`.
    constexpr Vec3(double x_, double y_) : x(x_), y(y_), z(0.0) {}
    /// Construct a fully three-dimensional vector.
    constexpr Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    /// Return componentwise vector addition.
    constexpr Vec3 operator+(const Vec3& other) const { return {x + other.x, y + other.y, z + other.z}; }
    /// Return componentwise vector subtraction.
    constexpr Vec3 operator-(const Vec3& other) const { return {x - other.x, y - other.y, z - other.z}; }
    /// Return the additive inverse.
    constexpr Vec3 operator-() const { return {-x, -y, -z}; }
    /// Return scalar multiplication.
    constexpr Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    /// Return scalar division.
    constexpr Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }

    /// Add another vector in place.
    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    /// Subtract another vector in place.
    Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    /// Multiply by a scalar in place.
    Vec3& operator*=(double s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    /// Divide by a scalar in place.
    Vec3& operator/=(double s) {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }
};

/// Compatibility alias for code paths that were originally planar.
using Vec2 = Vec3;

/// Return scalar-vector multiplication.
inline constexpr Vec3 operator*(double s, const Vec3& v) {
    return v * s;
}

/// Return the Euclidean dot product.
inline constexpr double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// Return the three-dimensional cross product.
inline constexpr Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

/// Return the squared Euclidean norm.
inline constexpr double norm_squared(const Vec3& v) {
    return dot(v, v);
}

/// Return the Euclidean norm.
inline double norm(const Vec3& v) {
    return std::sqrt(norm_squared(v));
}

}  // namespace fmmgalaxy

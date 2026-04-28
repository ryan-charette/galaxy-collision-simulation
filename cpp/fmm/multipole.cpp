#include "fmm/multipole.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace fmmgalaxy {

namespace {

constexpr int max_degree = 4;

using Exponent = std::array<int, 3>;
using Polynomial = std::array<double, 35>;

constexpr std::array<Exponent, 35> exponents{{
    {0, 0, 0},
    {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
    {2, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 2, 0}, {0, 1, 1}, {0, 0, 2},
    {3, 0, 0}, {2, 1, 0}, {2, 0, 1}, {1, 2, 0}, {1, 1, 1}, {1, 0, 2},
    {0, 3, 0}, {0, 2, 1}, {0, 1, 2}, {0, 0, 3},
    {4, 0, 0}, {3, 1, 0}, {3, 0, 1}, {2, 2, 0}, {2, 1, 1}, {2, 0, 2},
    {1, 3, 0}, {1, 2, 1}, {1, 1, 2}, {1, 0, 3}, {0, 4, 0}, {0, 3, 1},
    {0, 2, 2}, {0, 1, 3}, {0, 0, 4},
}};

int degree(const Exponent& exponent) {
    return exponent[0] + exponent[1] + exponent[2];
}

int index_of(int x, int y, int z) {
    for (std::size_t i = 0; i < exponents.size(); ++i) {
        if (exponents[i][0] == x && exponents[i][1] == y && exponents[i][2] == z) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

double pow_int(double value, int exponent) {
    double result = 1.0;
    for (int i = 0; i < exponent; ++i) {
        result *= value;
    }
    return result;
}

double monomial(const Vec2& value, const Exponent& exponent) {
    return pow_int(value.x, exponent[0]) *
           pow_int(value.y, exponent[1]) *
           pow_int(value.z, exponent[2]);
}

int binomial(int n, int k) {
    if (k < 0 || k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    if (n == 2) {
        return 2;
    }
    if (n == 3) {
        return k == 1 || k == 2 ? 3 : 1;
    }
    if (n == 4) {
        return k == 1 || k == 3 ? 4 : (k == 2 ? 6 : 1);
    }
    return 1;
}

double multinomial_shift(const Exponent& alpha, const Exponent& beta, const Vec2& shift) {
    if (beta[0] > alpha[0] || beta[1] > alpha[1] || beta[2] > alpha[2]) {
        return 0.0;
    }

    const int dx = alpha[0] - beta[0];
    const int dy = alpha[1] - beta[1];
    const int dz = alpha[2] - beta[2];
    return static_cast<double>(
               binomial(alpha[0], beta[0]) *
               binomial(alpha[1], beta[1]) *
               binomial(alpha[2], beta[2])
           ) *
           pow_int(shift.x, dx) *
           pow_int(shift.y, dy) *
           pow_int(shift.z, dz);
}

double moment_value(const CartesianMoments& moments, const Exponent& exponent, double mass) {
    const int d = degree(exponent);
    if (d == 0) {
        return mass;
    }
    if (d == 1) {
        return 0.0;
    }
    return moments.values[static_cast<std::size_t>(index_of(exponent[0], exponent[1], exponent[2]))];
}

Polynomial zero_polynomial() {
    Polynomial polynomial{};
    polynomial.fill(0.0);
    return polynomial;
}

Polynomial multiply(const Polynomial& a, const Polynomial& b) {
    Polynomial result = zero_polynomial();
    for (std::size_t i = 0; i < exponents.size(); ++i) {
        if (a[i] == 0.0) {
            continue;
        }
        for (std::size_t j = 0; j < exponents.size(); ++j) {
            if (b[j] == 0.0) {
                continue;
            }

            const Exponent exponent{
                exponents[i][0] + exponents[j][0],
                exponents[i][1] + exponents[j][1],
                exponents[i][2] + exponents[j][2],
            };
            if (degree(exponent) > max_degree) {
                continue;
            }

            const int index = index_of(exponent[0], exponent[1], exponent[2]);
            result[static_cast<std::size_t>(index)] += a[i] * b[j];
        }
    }
    return result;
}

Polynomial scale(const Polynomial& polynomial, double factor) {
    Polynomial result = polynomial;
    for (double& value : result) {
        value *= factor;
    }
    return result;
}

Polynomial add(const Polynomial& a, const Polynomial& b) {
    Polynomial result = a;
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] += b[i];
    }
    return result;
}

Polynomial component_polynomial(const Polynomial& inv_r3, int component, double component_value) {
    Polynomial result = scale(inv_r3, component_value);
    Exponent linear{0, 0, 0};
    linear[static_cast<std::size_t>(component)] = 1;
    Polynomial linear_poly = zero_polynomial();
    linear_poly[static_cast<std::size_t>(index_of(linear[0], linear[1], linear[2]))] = 1.0;
    return add(result, multiply(linear_poly, inv_r3));
}

Polynomial inv_r3_polynomial(const Vec2& delta, double softening) {
    const double h0 = norm_squared(delta) + softening * softening;
    const double base = std::pow(h0, -1.5);

    Polynomial q = zero_polynomial();
    q[static_cast<std::size_t>(index_of(1, 0, 0))] = 2.0 * delta.x / h0;
    q[static_cast<std::size_t>(index_of(0, 1, 0))] = 2.0 * delta.y / h0;
    q[static_cast<std::size_t>(index_of(0, 0, 1))] = 2.0 * delta.z / h0;
    q[static_cast<std::size_t>(index_of(2, 0, 0))] = 1.0 / h0;
    q[static_cast<std::size_t>(index_of(0, 2, 0))] = 1.0 / h0;
    q[static_cast<std::size_t>(index_of(0, 0, 2))] = 1.0 / h0;

    constexpr std::array<double, 5> coefficients{{1.0, -1.5, 1.875, -2.1875, 2.4609375}};
    Polynomial series = zero_polynomial();
    Polynomial power = zero_polynomial();
    power[0] = 1.0;

    for (int n = 0; n <= max_degree; ++n) {
        series = add(series, scale(power, coefficients[static_cast<std::size_t>(n)]));
        power = multiply(power, q);
    }

    return scale(series, base);
}

double expansion_moment_value(
    const CartesianMoments& moments,
    const Exponent& exponent,
    double mass,
    int expansion_order
) {
    const int d = degree(exponent);
    if (d == 0) {
        return mass;
    }
    if (d == 1 || d > expansion_order) {
        return 0.0;
    }
    return moments.values[static_cast<std::size_t>(index_of(exponent[0], exponent[1], exponent[2]))];
}

double evaluate_component(
    const Polynomial& polynomial,
    const CartesianMoments& moments,
    double mass,
    int expansion_order
) {
    double value = 0.0;
    for (std::size_t i = 0; i < exponents.size(); ++i) {
        if (degree(exponents[i]) > expansion_order) {
            continue;
        }
        value += polynomial[i] * expansion_moment_value(moments, exponents[i], mass, expansion_order);
    }
    return value;
}

}  // namespace

CartesianMoments zero_multipole_moments() {
    CartesianMoments moments;
    moments.values.fill(0.0);
    return moments;
}

void add_multipole_point(CartesianMoments& moments, const Vec2& offset, double mass) {
    for (std::size_t i = 0; i < exponents.size(); ++i) {
        const int d = degree(exponents[i]);
        if (d >= 2 && d <= max_degree) {
            moments.values[i] += mass * monomial(offset, exponents[i]);
        }
    }
}

void add_multipole_shifted_child(
    CartesianMoments& parent,
    const CartesianMoments& child,
    const Vec2& child_offset,
    double child_mass
) {
    for (std::size_t alpha_index = 0; alpha_index < exponents.size(); ++alpha_index) {
        const Exponent alpha = exponents[alpha_index];
        const int alpha_degree = degree(alpha);
        if (alpha_degree < 2 || alpha_degree > max_degree) {
            continue;
        }

        double value = 0.0;
        for (const Exponent& beta : exponents) {
            if (degree(beta) > alpha_degree) {
                continue;
            }
            const double shift_factor = multinomial_shift(alpha, beta, child_offset);
            if (shift_factor == 0.0) {
                continue;
            }
            value += shift_factor * moment_value(child, beta, child_mass);
        }
        parent.values[alpha_index] += value;
    }
}

Vec2 multipole_acceleration(
    const Vec2& target_position,
    const Vec2& source_center_of_mass,
    double source_mass,
    const CartesianMoments& source_moments,
    const PhysicsParams& params,
    int expansion_order
) {
    if (source_mass <= 0.0) {
        return {};
    }

    expansion_order = std::clamp(expansion_order, 0, max_degree);
    if (expansion_order == 1 || expansion_order == 3) {
        ++expansion_order;
    }

    const Vec2 delta = source_center_of_mass - target_position;
    const double h0 = norm_squared(delta) + params.softening * params.softening;
    if (h0 == 0.0) {
        return {};
    }

    const Polynomial inv = inv_r3_polynomial(delta, params.softening);
    const Polynomial gx = component_polynomial(inv, 0, delta.x);
    const Polynomial gy = component_polynomial(inv, 1, delta.y);
    const Polynomial gz = component_polynomial(inv, 2, delta.z);

    return {
        params.gravitational_constant * evaluate_component(gx, source_moments, source_mass, expansion_order),
        params.gravitational_constant * evaluate_component(gy, source_moments, source_mass, expansion_order),
        params.gravitational_constant * evaluate_component(gz, source_moments, source_mass, expansion_order),
    };
}

}  // namespace fmmgalaxy

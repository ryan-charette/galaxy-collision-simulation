#include "fmm/multipole.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace fmmgalaxy {

namespace {

constexpr int max_degree = 4;
constexpr std::size_t max_terms = 35;

using Exponent = std::array<int, 3>;
using Polynomial = std::array<double, max_terms>;
using Matrix = std::array<std::array<double, max_terms>, max_terms>;

constexpr std::array<Exponent, max_terms> exponents{{
    {0, 0, 0},
    {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
    {2, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 2, 0}, {0, 1, 1}, {0, 0, 2},
    {3, 0, 0}, {2, 1, 0}, {2, 0, 1}, {1, 2, 0}, {1, 1, 1}, {1, 0, 2},
    {0, 3, 0}, {0, 2, 1}, {0, 1, 2}, {0, 0, 3},
    {4, 0, 0}, {3, 1, 0}, {3, 0, 1}, {2, 2, 0}, {2, 1, 1}, {2, 0, 2},
    {1, 3, 0}, {1, 2, 1}, {1, 1, 2}, {1, 0, 3}, {0, 4, 0}, {0, 3, 1},
    {0, 2, 2}, {0, 1, 3}, {0, 0, 4},
}};

constexpr std::array<int, max_terms> exponent_degrees{{
    0,
    1, 1, 1,
    2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
}};

int coefficient_count(int expansion_order) {
    expansion_order = normalize_expansion_order(expansion_order);
    int count = 0;
    for (const int degree : exponent_degrees) {
        if (degree <= expansion_order) {
            ++count;
        }
    }
    return count;
}

int index_of(int x, int y, int z) {
    const int total = x + y + z;
    if (x < 0 || y < 0 || z < 0 || total > max_degree) {
        return -1;
    }

    const int degree_begin = total * (total + 1) * (total + 2) / 6;
    const int x_offset = total - x;
    const int before_x = x_offset * (x_offset + 1) / 2;
    const int before_y = (total - x) - y;
    return degree_begin + before_x + before_y;
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
    const int d = exponent[0] + exponent[1] + exponent[2];
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

Polynomial multiply(const Polynomial& a, const Polynomial& b, int expansion_order) {
    Polynomial result = zero_polynomial();
    for (std::size_t i = 0; i < exponents.size(); ++i) {
        if (a[i] == 0.0 || exponent_degrees[i] > expansion_order) {
            continue;
        }
        for (std::size_t j = 0; j < exponents.size(); ++j) {
            if (b[j] == 0.0 || exponent_degrees[j] > expansion_order) {
                continue;
            }

            const Exponent exponent{
                exponents[i][0] + exponents[j][0],
                exponents[i][1] + exponents[j][1],
                exponents[i][2] + exponents[j][2],
            };
            const int degree = exponent[0] + exponent[1] + exponent[2];
            if (degree > expansion_order) {
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

Polynomial component_polynomial(
    const Polynomial& inv_r3,
    int component,
    double component_value,
    int expansion_order
) {
    Polynomial result = scale(inv_r3, component_value);
    Exponent linear{0, 0, 0};
    linear[static_cast<std::size_t>(component)] = 1;
    Polynomial linear_poly = zero_polynomial();
    linear_poly[static_cast<std::size_t>(index_of(linear[0], linear[1], linear[2]))] = 1.0;
    return add(result, multiply(linear_poly, inv_r3, expansion_order));
}

Polynomial inv_r3_polynomial(const Vec2& delta, double softening, int expansion_order) {
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

    for (int n = 0; n <= expansion_order; ++n) {
        series = add(series, scale(power, coefficients[static_cast<std::size_t>(n)]));
        if (n == expansion_order) {
            break;
        }
        power = multiply(power, q, expansion_order);
    }

    return scale(series, base);
}

double expansion_moment_value(
    const CartesianMoments& moments,
    const Exponent& exponent,
    double mass,
    int expansion_order
) {
    const int d = exponent[0] + exponent[1] + exponent[2];
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
        if (exponent_degrees[i] > expansion_order) {
            continue;
        }
        value += polynomial[i] * expansion_moment_value(moments, exponents[i], mass, expansion_order);
    }
    return value;
}

Vec2 monopole_acceleration(
    const Vec2& target_position,
    const Vec2& source_position,
    double source_mass,
    const PhysicsParams& params
) {
    const Vec2 delta = source_position - target_position;
    const double s2 = norm_squared(delta) + params.softening * params.softening;
    if (s2 == 0.0 || source_mass <= 0.0) {
        return {};
    }
    const double inv_r = 1.0 / std::sqrt(s2);
    const double inv_r3 = inv_r * inv_r * inv_r;
    return delta * (params.gravitational_constant * source_mass * inv_r3);
}

struct InterpolationBasis {
    int order{0};
    int count{1};
    std::array<Vec2, max_terms> samples{};
    Matrix inverse{};
};

std::array<Vec2, max_terms> make_samples(int order) {
    std::array<Vec2, max_terms> samples{};
    if (order == 0) {
        samples[0] = {};
        return samples;
    }

    const double scale = 1.5 / static_cast<double>(order);
    const double center = static_cast<double>(order) / 3.0;
    int index = 0;
    for (int total = 0; total <= order; ++total) {
        for (int x = total; x >= 0; --x) {
            for (int y = total - x; y >= 0; --y) {
                const int z = total - x - y;
                samples[static_cast<std::size_t>(index++)] = {
                    (static_cast<double>(x) - center) * scale,
                    (static_cast<double>(y) - center) * scale,
                    (static_cast<double>(z) - center) * scale,
                };
            }
        }
    }
    return samples;
}

Matrix invert_vandermonde(const std::array<Vec2, max_terms>& samples, int count) {
    Matrix work{};
    Matrix inverse{};
    for (int row = 0; row < count; ++row) {
        for (int col = 0; col < count; ++col) {
            work[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
                monomial(samples[static_cast<std::size_t>(row)], exponents[static_cast<std::size_t>(col)]);
            inverse[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = row == col ? 1.0 : 0.0;
        }
    }

    for (int col = 0; col < count; ++col) {
        int pivot = col;
        double pivot_abs = std::abs(work[static_cast<std::size_t>(pivot)][static_cast<std::size_t>(col)]);
        for (int row = col + 1; row < count; ++row) {
            const double candidate =
                std::abs(work[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }

        if (pivot_abs < 1.0e-14) {
            throw std::runtime_error("local FMM interpolation basis is singular");
        }

        if (pivot != col) {
            std::swap(work[static_cast<std::size_t>(pivot)], work[static_cast<std::size_t>(col)]);
            std::swap(inverse[static_cast<std::size_t>(pivot)], inverse[static_cast<std::size_t>(col)]);
        }

        const double scale_factor = work[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)];
        for (int j = 0; j < count; ++j) {
            work[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)] /= scale_factor;
            inverse[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)] /= scale_factor;
        }

        for (int row = 0; row < count; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = work[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
            if (factor == 0.0) {
                continue;
            }
            for (int j = 0; j < count; ++j) {
                work[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)] -=
                    factor * work[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)];
                inverse[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)] -=
                    factor * inverse[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)];
            }
        }
    }

    return inverse;
}

InterpolationBasis make_basis(int order) {
    InterpolationBasis basis;
    basis.order = normalize_expansion_order(order);
    basis.count = coefficient_count(basis.order);
    basis.samples = make_samples(basis.order);
    basis.inverse = invert_vandermonde(basis.samples, basis.count);
    return basis;
}

const InterpolationBasis& basis_for_order(int order) {
    const int normalized = normalize_expansion_order(order);
    static const InterpolationBasis basis0 = make_basis(0);
    static const InterpolationBasis basis2 = make_basis(2);
    static const InterpolationBasis basis4 = make_basis(4);

    if (normalized <= 0) {
        return basis0;
    }
    if (normalized <= 2) {
        return basis2;
    }
    return basis4;
}

Vec2 scaled_offset(const Vec2& center, double radius, const Vec2& sample) {
    return {
        center.x + radius * sample.x,
        center.y + radius * sample.y,
        center.z + radius * sample.z,
    };
}

void add_sampled_values_to_local(
    LocalExpansion& target,
    const std::array<Vec2, max_terms>& values
) {
    const InterpolationBasis& basis = basis_for_order(target.order);
    for (int coefficient = 0; coefficient < basis.count; ++coefficient) {
        double ax = 0.0;
        double ay = 0.0;
        double az = 0.0;
        for (int sample = 0; sample < basis.count; ++sample) {
            const double factor =
                basis.inverse[static_cast<std::size_t>(coefficient)][static_cast<std::size_t>(sample)];
            ax += factor * values[static_cast<std::size_t>(sample)].x;
            ay += factor * values[static_cast<std::size_t>(sample)].y;
            az += factor * values[static_cast<std::size_t>(sample)].z;
        }
        target.ax[static_cast<std::size_t>(coefficient)] += ax;
        target.ay[static_cast<std::size_t>(coefficient)] += ay;
        target.az[static_cast<std::size_t>(coefficient)] += az;
    }
}

}  // namespace

int normalize_expansion_order(int expansion_order) {
    expansion_order = std::clamp(expansion_order, 0, max_degree);
    if (expansion_order == 1) {
        return 2;
    }
    if (expansion_order == 3) {
        return 4;
    }
    return expansion_order;
}

CartesianMoments zero_multipole_moments() {
    CartesianMoments moments;
    moments.values.fill(0.0);
    return moments;
}

LocalExpansion zero_local_expansion(const Vec2& center, double radius, int expansion_order) {
    LocalExpansion local;
    local.center = center;
    local.radius = std::max(radius, 1.0e-12);
    local.order = normalize_expansion_order(expansion_order);
    local.ax.fill(0.0);
    local.ay.fill(0.0);
    local.az.fill(0.0);
    return local;
}

void add_multipole_point(CartesianMoments& moments, const Vec2& offset, double mass) {
    for (std::size_t i = 0; i < exponents.size(); ++i) {
        const int d = exponent_degrees[i];
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
        const int alpha_degree = exponent_degrees[alpha_index];
        if (alpha_degree < 2 || alpha_degree > max_degree) {
            continue;
        }

        double value = 0.0;
        for (const Exponent& beta : exponents) {
            const int beta_degree = beta[0] + beta[1] + beta[2];
            if (beta_degree > alpha_degree) {
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

    expansion_order = normalize_expansion_order(expansion_order);
    if (expansion_order <= 0) {
        return monopole_acceleration(target_position, source_center_of_mass, source_mass, params);
    }

    const Vec2 delta = source_center_of_mass - target_position;
    const double h0 = norm_squared(delta) + params.softening * params.softening;
    if (h0 == 0.0) {
        return {};
    }

    const Polynomial inv = inv_r3_polynomial(delta, params.softening, expansion_order);
    const Polynomial gx = component_polynomial(inv, 0, delta.x, expansion_order);
    const Polynomial gy = component_polynomial(inv, 1, delta.y, expansion_order);
    const Polynomial gz = component_polynomial(inv, 2, delta.z, expansion_order);

    return {
        params.gravitational_constant * evaluate_component(gx, source_moments, source_mass, expansion_order),
        params.gravitational_constant * evaluate_component(gy, source_moments, source_mass, expansion_order),
        params.gravitational_constant * evaluate_component(gz, source_moments, source_mass, expansion_order),
    };
}

void add_multipole_to_local(
    LocalExpansion& target,
    const Vec2& source_center_of_mass,
    double source_mass,
    const CartesianMoments& source_moments,
    const PhysicsParams& params
) {
    if (source_mass <= 0.0) {
        return;
    }

    const InterpolationBasis& basis = basis_for_order(target.order);
    std::array<Vec2, max_terms> values{};
    for (int sample = 0; sample < basis.count; ++sample) {
        const Vec2 position = scaled_offset(
            target.center,
            target.radius,
            basis.samples[static_cast<std::size_t>(sample)]
        );
        values[static_cast<std::size_t>(sample)] = multipole_acceleration(
            position,
            source_center_of_mass,
            source_mass,
            source_moments,
            params,
            target.order
        );
    }
    add_sampled_values_to_local(target, values);
}

void add_local_to_local(LocalExpansion& target, const LocalExpansion& source) {
    const InterpolationBasis& basis = basis_for_order(target.order);
    std::array<Vec2, max_terms> values{};
    for (int sample = 0; sample < basis.count; ++sample) {
        const Vec2 position = scaled_offset(
            target.center,
            target.radius,
            basis.samples[static_cast<std::size_t>(sample)]
        );
        values[static_cast<std::size_t>(sample)] = evaluate_local_acceleration(source, position);
    }
    add_sampled_values_to_local(target, values);
}

Vec2 evaluate_local_acceleration(const LocalExpansion& local, const Vec2& target_position) {
    const double inv_radius = local.radius > 0.0 ? 1.0 / local.radius : 1.0;
    const Vec2 offset{
        (target_position.x - local.center.x) * inv_radius,
        (target_position.y - local.center.y) * inv_radius,
        (target_position.z - local.center.z) * inv_radius,
    };
    const int order = normalize_expansion_order(local.order);
    const int count = coefficient_count(order);

    Vec2 acceleration{};
    for (int i = 0; i < count; ++i) {
        const double basis_value = monomial(offset, exponents[static_cast<std::size_t>(i)]);
        acceleration.x += local.ax[static_cast<std::size_t>(i)] * basis_value;
        acceleration.y += local.ay[static_cast<std::size_t>(i)] * basis_value;
        acceleration.z += local.az[static_cast<std::size_t>(i)] * basis_value;
    }
    return acceleration;
}

}  // namespace fmmgalaxy

#include "core/config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace fmmgalaxy {

namespace {

std::string trim(std::string value) {
    const auto is_not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), is_not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), is_not_space).base(), value.end());
    return value;
}

std::string strip_comment(const std::string& line) {
    bool in_string = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '"') {
            in_string = !in_string;
        }
        if (!in_string && line[i] == '#') {
            return line.substr(0, i);
        }
    }
    return line;
}

std::string unquote(std::string value) {
    value = trim(std::move(value));
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

std::vector<double> parse_number_array(std::string value) {
    value = trim(std::move(value));
    if (value.size() < 2 || value.front() != '[' || value.back() != ']') {
        throw std::runtime_error("Expected numeric array value, got: " + value);
    }

    value = value.substr(1, value.size() - 2);
    std::vector<double> numbers;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            numbers.push_back(std::stod(item));
        }
    }
    return numbers;
}

Vec2 parse_vec2(const std::string& value) {
    const auto numbers = parse_number_array(value);
    if (numbers.size() != 2) {
        throw std::runtime_error("Expected [x, y] vector value, got: " + value);
    }
    return {numbers[0], numbers[1]};
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

void set_simulation_value(SimulationConfig& config, const std::string& key, const std::string& value) {
    if (key == "name") {
        config.name = unquote(value);
    } else if (key == "solver") {
        config.solver = lowercase(unquote(value));
    } else if (key == "seed") {
        config.seed = static_cast<std::uint64_t>(std::stoull(value));
    } else if (key == "n_particles") {
        config.n_particles = static_cast<std::size_t>(std::stoull(value));
    } else if (key == "steps") {
        config.steps = std::stoi(value);
    } else if (key == "dt") {
        config.dt = std::stod(value);
    } else if (key == "snapshot_every") {
        config.snapshot_every = std::stoi(value);
    } else if (key == "theta" || key == "tree_theta") {
        config.tree_theta = std::stod(value);
    } else if (key == "leaf_capacity" || key == "tree_leaf_capacity") {
        config.tree_leaf_capacity = static_cast<std::size_t>(std::stoull(value));
    } else if (key == "expansion_order" || key == "fmm_expansion_order") {
        config.fmm_expansion_order = std::stoi(value);
    }
}

void set_physics_value(SimulationConfig& config, const std::string& key, const std::string& value) {
    if (key == "G" || key == "gravitational_constant") {
        config.physics.gravitational_constant = std::stod(value);
    } else if (key == "softening") {
        config.physics.softening = std::stod(value);
    }
}

void set_output_value(SimulationConfig& config, const std::string& key, const std::string& value) {
    if (key == "directory") {
        config.output.directory = unquote(value);
    } else if (key == "format") {
        config.output.format = lowercase(unquote(value));
    }
}

void set_galaxy_value(GalaxyConfig& galaxy, const std::string& key, const std::string& value) {
    if (key == "n_particles") {
        galaxy.n_particles = static_cast<std::size_t>(std::stoull(value));
    } else if (key == "mass") {
        galaxy.mass = std::stod(value);
    } else if (key == "radius") {
        galaxy.radius = std::stod(value);
    } else if (key == "position") {
        galaxy.position = parse_vec2(value);
    } else if (key == "velocity") {
        galaxy.velocity = parse_vec2(value);
    } else if (key == "orientation") {
        galaxy.orientation = std::stod(value);
    } else if (key == "group_id") {
        galaxy.group_id = static_cast<std::uint32_t>(std::stoul(value));
    }
}

}  // namespace

SimulationConfig default_config() {
    SimulationConfig config;
    config.galaxies = {
        GalaxyConfig{128, 1.0, 1.0, {-0.8, 0.0}, {0.15, 0.0}, 0.0, 0},
        GalaxyConfig{128, 1.0, 1.0, {0.8, 0.0}, {-0.15, 0.0}, 3.141592653589793, 1},
    };
    config.n_particles = 256;
    return config;
}

SimulationConfig load_config(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open config file: " + path.string());
    }

    SimulationConfig config = default_config();
    config.galaxies.clear();

    std::string section;
    std::map<std::string, GalaxyConfig> galaxies_by_name;
    std::vector<std::string> galaxy_order;

    std::string raw_line;
    int line_number = 0;
    while (std::getline(input, raw_line)) {
        ++line_number;
        std::string line = trim(strip_comment(raw_line));
        if (line.empty()) {
            continue;
        }

        if (line.front() == '[' && line.back() == ']') {
            section = line.substr(1, line.size() - 2);
            if (section.rfind("galaxy.", 0) == 0) {
                const std::string name = section.substr(std::string("galaxy.").size());
                if (galaxies_by_name.find(name) == galaxies_by_name.end()) {
                    galaxies_by_name[name] = GalaxyConfig{};
                    galaxy_order.push_back(name);
                }
            }
            continue;
        }

        const auto equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error("Invalid config line " + std::to_string(line_number) + ": " + raw_line);
        }

        const std::string key = trim(line.substr(0, equals));
        const std::string value = trim(line.substr(equals + 1));

        if (section == "simulation") {
            set_simulation_value(config, key, value);
        } else if (section == "physics") {
            set_physics_value(config, key, value);
        } else if (section == "output") {
            set_output_value(config, key, value);
        } else if (section == "tree" || section == "fmm") {
            set_simulation_value(config, key, value);
        } else if (section.rfind("galaxy.", 0) == 0) {
            const std::string name = section.substr(std::string("galaxy.").size());
            set_galaxy_value(galaxies_by_name[name], key, value);
        }
    }

    for (const auto& name : galaxy_order) {
        config.galaxies.push_back(galaxies_by_name[name]);
    }

    if (config.galaxies.empty()) {
        config = default_config();
    }

    if (config.snapshot_every <= 0) {
        throw std::runtime_error("snapshot_every must be positive");
    }
    if (config.steps < 0) {
        throw std::runtime_error("steps must be non-negative");
    }
    if (config.dt <= 0.0) {
        throw std::runtime_error("dt must be positive");
    }
    if (config.physics.softening < 0.0) {
        throw std::runtime_error("softening must be non-negative");
    }

    return config;
}

}  // namespace fmmgalaxy

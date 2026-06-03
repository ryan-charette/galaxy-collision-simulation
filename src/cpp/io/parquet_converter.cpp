#include "io/parquet_converter.hpp"

#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace fmmgalaxy {

namespace {

std::string quote_command_arg(const std::string& value) {
    std::string quoted = "\"";
    for (const char ch : value) {
        if (ch == '"') {
            quoted.push_back('\\');
        }
        quoted.push_back(ch);
    }
    quoted.push_back('"');
    return quoted;
}

std::vector<std::string> python_commands() {
    std::vector<std::string> commands;
    if (const char* configured_python = std::getenv("FMM_GALAXY_PYTHON")) {
        commands.emplace_back(quote_command_arg(configured_python));
    }
    commands.emplace_back("python");
    commands.emplace_back("python3");
    commands.emplace_back("py -3");
    return commands;
}

}  // namespace

bool ParquetConverter::convert_snapshot_csv(
    const std::filesystem::path& csv_path,
    const std::filesystem::path& parquet_path,
    int step,
    double time
) const {
    for (const auto& python : python_commands()) {
        std::ostringstream command;
        command << python
                << " -m python.utils.parquet_io"
                << " --input " << quote_command_arg(csv_path.string())
                << " --output " << quote_command_arg(parquet_path.string())
                << " --step " << step
                << " --time " << std::setprecision(17) << time;
        if (std::system(command.str().c_str()) == 0) {
            return true;
        }
    }
    return false;
}

}  // namespace fmmgalaxy

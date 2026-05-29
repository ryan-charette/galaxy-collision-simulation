#pragma once

#include <filesystem>

namespace fmmgalaxy {

class ParquetConverter {
public:
    bool convert_snapshot_csv(
        const std::filesystem::path& csv_path,
        const std::filesystem::path& parquet_path,
        int step,
        double time
    ) const;
};

}  // namespace fmmgalaxy

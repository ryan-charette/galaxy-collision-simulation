#pragma once

#include <filesystem>

namespace fmmgalaxy {

/// Thin C++ wrapper around the Python CSV-to-Parquet snapshot converter.
class ParquetConverter {
public:
    /// Convert a CSV snapshot to Parquet, returning `false` if conversion fails.
    bool convert_snapshot_csv(
        const std::filesystem::path& csv_path,
        const std::filesystem::path& parquet_path,
        int step,
        double time
    ) const;
};

}  // namespace fmmgalaxy

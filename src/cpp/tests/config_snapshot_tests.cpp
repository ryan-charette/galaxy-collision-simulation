#include "core/config.hpp"
#include "core/diagnostics.hpp"
#include "core/initial_conditions.hpp"
#include "core/provenance.hpp"
#include "io/snapshot_writer.hpp"
#include "mpi/distributed_solver.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

TEST_CASE("Config parsing and snapshot writing preserve run metadata", "[config][io]") {
    using Catch::Matchers::WithinAbs;
    const std::filesystem::path test_config_path = "test_config.toml";
    const std::filesystem::path output_dir = "test_output";

    std::filesystem::remove(test_config_path);
    std::filesystem::remove_all(output_dir);

    std::ofstream config_file(test_config_path, std::ios::trunc);
    config_file << "[simulation]\nname=\"unit\"\nsolver=\"tree\"\ndim=3\nsteps=2\ndt=0.01\nsnapshot_every=1\n"
                << "[physics]\nG=1.0\nsoftening=0.02\n"
                << "[galaxy.primary]\nn_particles=4\nmass=1.0\nradius=1.0\n"
                << "position=[0.0,0.0,0.1]\nvelocity=[0.0,0.0,0.0]\norientation=0.0\ngroup_id=3\n"
                << "thickness=0.05\ninclination=0.2\n"
                << "[output]\ndirectory=\"test_output\"\nformat=\"csv\"\nacceleration_dump=true\n";
    config_file.close();

    const auto loaded = fmmgalaxy::load_config(test_config_path);
    CHECK(loaded.name == "unit");
    CHECK(loaded.dim == 3);
    REQUIRE(loaded.galaxies.size() == 1);
    CHECK(loaded.galaxies[0].group_id == 3);
    CHECK_THAT(loaded.galaxies[0].position.z, WithinAbs(0.1, 1.0e-12));
    CHECK(loaded.output.format == fmmgalaxy::OutputFormat::Csv);
    CHECK(loaded.output.acceleration_dump);
    CHECK(fmmgalaxy::parse_output_format("parquet") == fmmgalaxy::OutputFormat::Parquet);

    const auto generated =
        fmmgalaxy::generate_galaxies(loaded.galaxies, loaded.physics, loaded.seed);
    const auto diagnostics = fmmgalaxy::compute_diagnostics(generated, loaded.physics);

    fmmgalaxy::SnapshotWriter writer(loaded);
    const fmmgalaxy::MpiExecution serial_execution{};
    const auto provenance = fmmgalaxy::collect_run_provenance(&test_config_path, serial_execution);
    CHECK(provenance.config_sha256.size() == 64);
    writer.write_metadata(loaded, generated.size(), provenance);
    writer.write_snapshot(0, 0.0, generated);
    writer.write_accelerations(0, 0.0, generated);
    writer.write_diagnostics(0, 0.0, diagnostics, generated.size());
    CHECK(std::filesystem::exists(output_dir / "snapshot_000000.csv"));
    CHECK(std::filesystem::exists(output_dir / "accelerations_000000.csv"));
    CHECK(std::filesystem::exists(output_dir / "diagnostics.csv"));

    std::ifstream metadata_file(output_dir / "metadata.json");
    const std::string metadata_json(
        (std::istreambuf_iterator<char>(metadata_file)),
        std::istreambuf_iterator<char>()
    );
    CHECK(metadata_json.find("\"git_commit\"") != std::string::npos);
    CHECK(metadata_json.find("\"config_sha256\"") != std::string::npos);
}

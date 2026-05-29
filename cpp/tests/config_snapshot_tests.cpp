#include "tests/test_support.hpp"

#include "core/config.hpp"
#include "core/diagnostics.hpp"
#include "core/initial_conditions.hpp"
#include "core/provenance.hpp"
#include "io/snapshot_writer.hpp"
#include "mpi/distributed_solver.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

int run_config_snapshot_tests() {
    using fmmgalaxy::tests::near;
    using fmmgalaxy::tests::require;

    int failures = 0;

    std::ofstream config_file("test_config.toml", std::ios::trunc);
    config_file << "[simulation]\nname=\"unit\"\nsolver=\"tree\"\ndim=3\nsteps=2\ndt=0.01\nsnapshot_every=1\n"
                << "[physics]\nG=1.0\nsoftening=0.02\n"
                << "[galaxy.primary]\nn_particles=4\nmass=1.0\nradius=1.0\n"
                << "position=[0.0,0.0,0.1]\nvelocity=[0.0,0.0,0.0]\norientation=0.0\ngroup_id=3\n"
                << "thickness=0.05\ninclination=0.2\n"
                << "[output]\ndirectory=\"test_output\"\nformat=\"csv\"\nacceleration_dump=true\n";
    config_file.close();

    const std::filesystem::path test_config_path = "test_config.toml";
    const auto loaded = fmmgalaxy::load_config(test_config_path);
    failures += !require(loaded.name == "unit", "config parser reads simulation name");
    failures += !require(loaded.dim == 3, "config parser reads 3D dimension");
    failures += !require(loaded.galaxies.size() == 1, "config parser reads galaxy section");
    failures += !require(loaded.galaxies[0].group_id == 3, "config parser reads group id");
    failures += !require(
        near(loaded.galaxies[0].position.z, 0.1, 1.0e-12),
        "config parser reads z position"
    );
    failures += !require(
        loaded.output.format == fmmgalaxy::OutputFormat::Csv,
        "config parser reads csv output format"
    );
    failures += !require(loaded.output.acceleration_dump, "config parser reads acceleration dump flag");
    failures += !require(
        fmmgalaxy::parse_output_format("parquet") == fmmgalaxy::OutputFormat::Parquet,
        "config parser accepts parquet output format"
    );

    const auto generated =
        fmmgalaxy::generate_galaxies(loaded.galaxies, loaded.physics, loaded.seed);
    const auto diagnostics = fmmgalaxy::compute_diagnostics(generated, loaded.physics);

    fmmgalaxy::SnapshotWriter writer(loaded);
    const fmmgalaxy::MpiExecution serial_execution{};
    const auto provenance = fmmgalaxy::collect_run_provenance(&test_config_path, serial_execution);
    failures += !require(provenance.config_sha256.size() == 64, "provenance hashes config file");
    writer.write_metadata(loaded, generated.size(), provenance);
    writer.write_snapshot(0, 0.0, generated);
    writer.write_accelerations(0, 0.0, generated);
    writer.write_diagnostics(0, 0.0, diagnostics, generated.size());
    failures += !require(std::filesystem::exists("test_output/snapshot_000000.csv"), "snapshot writer creates csv");
    failures += !require(
        std::filesystem::exists("test_output/accelerations_000000.csv"),
        "snapshot writer creates acceleration dump"
    );
    failures += !require(std::filesystem::exists("test_output/diagnostics.csv"), "snapshot writer creates diagnostics");

    std::ifstream metadata_file("test_output/metadata.json");
    const std::string metadata_json(
        (std::istreambuf_iterator<char>(metadata_file)),
        std::istreambuf_iterator<char>()
    );
    failures += !require(
        metadata_json.find("\"git_commit\"") != std::string::npos,
        "metadata includes git commit"
    );
    failures += !require(
        metadata_json.find("\"config_sha256\"") != std::string::npos,
        "metadata includes config hash"
    );

    return failures;
}

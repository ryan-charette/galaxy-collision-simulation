#include "core/provenance.hpp"

#include "build_config.hpp"
#include "cuda/cuda_solver.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

namespace fmmgalaxy {

namespace {

constexpr std::array<std::uint32_t, 64> sha256_k = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U,
};

std::uint32_t rotr(std::uint32_t value, int bits) {
    return (value >> bits) | (value << (32 - bits));
}

class Sha256 {
public:
    void update(const unsigned char* data, std::size_t size) {
        bit_count_ += static_cast<std::uint64_t>(size) * 8U;
        for (std::size_t i = 0; i < size; ++i) {
            buffer_[buffer_size_++] = data[i];
            if (buffer_size_ == buffer_.size()) {
                transform(buffer_.data());
                buffer_size_ = 0;
            }
        }
    }

    std::string final_hex() {
        buffer_[buffer_size_++] = 0x80U;
        if (buffer_size_ > 56) {
            while (buffer_size_ < 64) {
                buffer_[buffer_size_++] = 0U;
            }
            transform(buffer_.data());
            buffer_size_ = 0;
        }
        while (buffer_size_ < 56) {
            buffer_[buffer_size_++] = 0U;
        }
        for (int shift = 56; shift >= 0; shift -= 8) {
            buffer_[buffer_size_++] = static_cast<unsigned char>((bit_count_ >> shift) & 0xffU);
        }
        transform(buffer_.data());

        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (const auto value : state_) {
            out << std::setw(8) << value;
        }
        return out.str();
    }

private:
    void transform(const unsigned char* chunk) {
        std::array<std::uint32_t, 64> w{};
        for (int i = 0; i < 16; ++i) {
            const int offset = i * 4;
            w[static_cast<std::size_t>(i)] =
                (static_cast<std::uint32_t>(chunk[offset]) << 24) |
                (static_cast<std::uint32_t>(chunk[offset + 1]) << 16) |
                (static_cast<std::uint32_t>(chunk[offset + 2]) << 8) |
                static_cast<std::uint32_t>(chunk[offset + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[static_cast<std::size_t>(i - 15)], 7) ^
                                     rotr(w[static_cast<std::size_t>(i - 15)], 18) ^
                                     (w[static_cast<std::size_t>(i - 15)] >> 3);
            const std::uint32_t s1 = rotr(w[static_cast<std::size_t>(i - 2)], 17) ^
                                     rotr(w[static_cast<std::size_t>(i - 2)], 19) ^
                                     (w[static_cast<std::size_t>(i - 2)] >> 10);
            w[static_cast<std::size_t>(i)] =
                w[static_cast<std::size_t>(i - 16)] + s0 +
                w[static_cast<std::size_t>(i - 7)] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];

        for (int i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 =
                h + s1 + ch + sha256_k[static_cast<std::size_t>(i)] + w[static_cast<std::size_t>(i)];
            const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{
        0x6a09e667U,
        0xbb67ae85U,
        0x3c6ef372U,
        0xa54ff53aU,
        0x510e527fU,
        0x9b05688cU,
        0x1f83d9abU,
        0x5be0cd19U,
    };
    std::array<unsigned char, 64> buffer_{};
    std::size_t buffer_size_{0};
    std::uint64_t bit_count_{0};
};

std::string trim(std::string value) {
    while (!value.empty() && (value.back() == '\n' || value.back() == '\r' || value.back() == ' ')) {
        value.pop_back();
    }
    while (!value.empty() && value.front() == ' ') {
        value.erase(value.begin());
    }
    return value;
}

std::string command_output(const std::string& command) {
#ifdef _WIN32
    const std::string redirected = command + " 2>NUL";
#else
    const std::string redirected = command + " 2>/dev/null";
#endif
    std::array<char, 256> buffer{};
    std::string output;
    FILE* pipe = popen(redirected.c_str(), "r");
    if (!pipe) {
        return "";
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }
    const int status = pclose(pipe);
    if (status != 0) {
        return "";
    }
    return trim(output);
}

std::string build_type() {
#ifdef NDEBUG
    return "Release";
#else
    return "Debug";
#endif
}

std::string hostname() {
    if (const char* computer_name = std::getenv("COMPUTERNAME")) {
        if (std::strlen(computer_name) > 0) {
            return computer_name;
        }
    }
    if (const char* host = std::getenv("HOSTNAME")) {
        if (std::strlen(host) > 0) {
            return host;
        }
    }
    return "unknown";
}

std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

}  // namespace

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open file for SHA-256 hashing: " + path.string());
    }

    Sha256 hasher;
    std::array<char, 32768> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize read_count = input.gcount();
        if (read_count > 0) {
            hasher.update(
                reinterpret_cast<const unsigned char*>(buffer.data()),
                static_cast<std::size_t>(read_count)
            );
        }
    }
    return hasher.final_hex();
}

RunProvenance collect_run_provenance(
    const std::filesystem::path* config_path,
    const MpiExecution& execution
) {
    RunProvenance provenance;
    provenance.git_commit = command_output("git rev-parse HEAD");
    if (provenance.git_commit.empty()) {
        provenance.git_commit = "unavailable";
    }
    provenance.git_branch = command_output("git rev-parse --abbrev-ref HEAD");
    if (provenance.git_branch.empty()) {
        provenance.git_branch = "unavailable";
    }
    provenance.git_dirty = !command_output("git status --porcelain").empty();
    provenance.build_type = build_type();
    provenance.compiler = FMM_GALAXY_COMPILER_ID;
    provenance.compiler_version = FMM_GALAXY_COMPILER_VERSION;
    provenance.cmake_enable_mpi = FMM_GALAXY_CMAKE_ENABLE_MPI != 0;
    provenance.cmake_enable_cuda = FMM_GALAXY_CMAKE_ENABLE_CUDA != 0;
    provenance.cuda_available = cuda_solver_available();
    provenance.cuda_device_name = cuda_device_name();
    provenance.mpi_enabled = execution.enabled;
    provenance.rank_count = execution.size;
    provenance.hostname = hostname();
    provenance.timestamp_utc = utc_timestamp();

    if (config_path != nullptr) {
        provenance.config_path = std::filesystem::absolute(*config_path).string();
        provenance.config_sha256 = sha256_file(*config_path);
    }

    return provenance;
}

}  // namespace fmmgalaxy

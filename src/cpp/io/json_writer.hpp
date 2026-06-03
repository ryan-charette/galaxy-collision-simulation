#pragma once

#include <ostream>
#include <string>

namespace fmmgalaxy {

inline std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        if (ch == '"' || ch == '\\') {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    return escaped;
}

inline const char* json_bool(bool value) {
    return value ? "true" : "false";
}

inline void write_json_string(std::ostream& output, const std::string& value) {
    output << '"' << json_escape(value) << '"';
}

}  // namespace fmmgalaxy

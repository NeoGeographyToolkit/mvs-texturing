#pragma once

#include <sstream>
#include <vector>
#include <string>

template <typename T>
const std::vector<std::string> choice_strings();

/** Enum representing a data term. */
enum DataTerm {
    AREA = 0,
    GMI = 1
};
template <> inline
const std::vector<std::string> choice_strings<DataTerm>() {
    return {"area", "gmi"};
}

/** Enum representing a smoothness term. */
enum SmoothnessTerm {
    POTTS = 0,
    EDI = 1
};
template <> inline
const std::vector<std::string> choice_strings<SmoothnessTerm>() {
    return {"potts", "edi"};
}

/** Enum representing the choice of outlier removal. */
enum OutlierRemoval {
    NONE = 0,
    GAUSS_DAMPING = 1,
    GAUSS_CLAMPING = 2
};
template <> inline
const std::vector<std::string> choice_strings<OutlierRemoval>() {
    return {"none", "gauss_damping", "gauss_clamping"};
}

template <typename T> inline
const std::string choice_string(T i) {
    return choice_strings<T>()[static_cast<std::size_t>(i)];
}

template <typename T> inline
T parse_choice(std::string s) {
    const std::vector<std::string> strings = choice_strings<T>();
    const std::size_t n = strings.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (s == strings[i]) {
            return static_cast<T>(i);
        }
    }

    std::stringstream ss;
    ss << "Invalid choice: " << s << " (Available choices:";
    for (std::size_t i = 0; i < n; ++i) {
        ss << " " << strings[i];
        if(i != n - 1) ss << ",";
    }
    ss << ")";

    throw std::invalid_argument(ss.str());
}

//TEX_NAMESPACE_BEGIN
struct Settings {
    bool verbose;

    DataTerm data_term;
    SmoothnessTerm smoothness_term;
    OutlierRemoval outlier_removal;

    bool geometric_visibility_test;
    bool global_seam_leveling;
    bool local_seam_leveling;
};
//TEX_NAMESPACE_END


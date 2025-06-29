#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>

// Helper function to trim both ends of a string
static inline std::string trim(const std::string& s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) start++;
    auto end = s.end();
    do {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(*end));
    return std::string(start, end + 1);
}

std::string readFastaSequence(const std::string& fasta_path) {
    std::ifstream infile(fasta_path);
    if (!infile) {
        throw std::runtime_error("Could not open FASTA file: " + fasta_path);
    }

    std::string line, sequence;
    bool header_skipped = false;

    while (std::getline(infile, line)) {
        if (!header_skipped) {
            if (!line.empty() && line[0] == '>') {
                header_skipped = true;
                continue;
            }
        }
        if (!line.empty() && line[0] != '>') {
            sequence += trim(line);
        }
    }
    return sequence;
}

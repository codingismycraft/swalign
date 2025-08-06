#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <random>
#include <cctype>

#define RANDOM_NAME_LENGTH 8

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


std::string generate_random_name(){
    const std::string chars =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, chars.size() - 1);

    std::string name;
    for (int i = 0; i < RANDOM_NAME_LENGTH; ++i) {
        name += chars[distrib(gen)];
    }
    return name + "-bigarray.bin";
}

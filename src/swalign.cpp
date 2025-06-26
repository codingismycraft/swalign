/*
 **************************************************************************
 *
 * Copyright (c) 2025 John Pazarzis
 *
 * Licensed under the GPL License.
 *
 * Program Name: swalign
 *
 * The solution is based on the Smith-Waterman algorithm, which is a dynamic
 * programming algorithm for performing local sequence alignment.
 *
 * This program (swalign) computes the optimal local alignment(s) between
 * two biological (or arbitrary) sequences using the Smith-Waterman
 * algorithm. Provide your sequences as files, select your scoring
 * scheme, and discover the best local similarities within your data!
 *
 * Usage:
 *    swalign <seq1_file> <seq2_file> [options]
 *
 * Arguments:
 *    <seq1_file>    File containing the first sequence (required)
 *    <seq2_file>    File containing the second sequence (required)
 *
 * Options:
 *    -m <int>       Match score (default: 2)
 *    -x <int>       Mismatch penalty (default: -1)
 *    -g <int>       Gap penalty (default: -1)
 *    -h, --help     Print this help message and exit
 *
 * Notes:
 *    - All score and penalty values must be integers between -10 and 10.
 *    - Files must exist and be readable.
 *
 * Example:
 *    ./swalign seq1.txt seq2.txt -m 2 -x -1 -g -1
 *
 **************************************************************************
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include "score_matrix.h"

// Prints help message for command-line usage
void print_help(const char* progname) {
    std::cout <<
        "Usage: " << progname << " <seq1_file> <seq2_file> [options]\n"
        "Compute highest scoring local sequence alignments using the Smith-Waterman algorithm.\n"
        "\n"
        "Arguments:\n"
        "  <seq1_file>            File containing the first sequence (required)\n"
        "  <seq2_file>            File containing the second sequence (required)\n"
        "\n"
        "Options:\n"
        "  -m <int>               Match score (default: 2)\n"
        "  -x <int>               Mismatch penalty (default: -1)\n"
        "  -g <int>               Gap penalty (default: -1)\n"
        "  -h, --help             Print this help message and exit\n"
        "\n"
        "Notes:\n"
        "  - All score and penalty values must be integers between -10 and 10.\n"
        "  - Files must exist and be readable.\n"
        << std::endl;
}

// Utility function to read sequence from a text file, skipping whitespace
std::string read_sequence_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: File '" << filename << "' does not exist or cannot be opened." << std::endl;
        exit(1);
    }
    std::string sequence, line;
    while (std::getline(file, line)) {
        for (char c : line) {
            if (!isspace(static_cast<unsigned char>(c))) {
                sequence += c;
            }
        }
    }
    return sequence;
}

// Clamp a value between min and max inclusive
int clamp(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

int main(int argc, char* argv[]) {
    // Print help if requested or insufficient arguments
    if (argc < 3) {
        print_help(argv[0]);
        return 1;
    }
    if (argc >= 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        print_help(argv[0]);
        return 0;
    }

    std::string seq1_file = argv[1];
    std::string seq2_file = argv[2];

    // Check if both files exist and can be opened
    std::ifstream f1(seq1_file);
    if (!f1) {
        std::cerr << "Error: File '" << seq1_file << "' does not exist or cannot be opened." << std::endl;
        return 1;
    }
    std::ifstream f2(seq2_file);
    if (!f2) {
        std::cerr << "Error: File '" << seq2_file << "' does not exist or cannot be opened." << std::endl;
        return 1;
    }

    // Default scoring parameters
    int match_score = 2;
    int mismatch_penalty = -1;
    int gap_penalty = -1;

    // Parse command-line options
    for (int i = 3; i < argc; ++i) {
        if ((strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)) {
            print_help(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            match_score = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            mismatch_penalty = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
            gap_penalty = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown or incomplete option: " << argv[i] << "\n";
            print_help(argv[0]);
            return 1;
        }
    }

    // Validate scoring parameters
    if (match_score < -10 || match_score > 10 ||
        mismatch_penalty < -10 || mismatch_penalty > 10 ||
        gap_penalty < -10 || gap_penalty > 10) {
        std::cerr << "Error: Scores and penalties must be integers between -10 and 10.\n";
        return 1;
    }

    // Read sequences from files, asserting file existence
    std::string seq1 = read_sequence_from_file(seq1_file);
    std::string seq2 = read_sequence_from_file(seq2_file);

    // Create ScoreMatrix instance and compute local alignments
    ScoreMatrix score_matrix(seq1, seq2, match_score, mismatch_penalty, gap_penalty);
    auto alignments = score_matrix.getLocalAlignment();

    // Output the result(s)
    if (alignments.empty()) {
        std::cout << "No local alignments found.\n";
    } else {
        std::cout << "Highest scoring local alignment(s):\n";
        for (const auto& alignment : alignments) {
            std::cout << alignment << std::endl;
        }
    }

    return 0;
}

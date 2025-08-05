#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "local_alignment.h"

#define DEFAULT_MATCH_SCORE 2
#define DEFAULT_MISMATCH_PENALTY -1
#define DEFAULT_GAP_PENALTY -1


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
        if (line.empty() || line[0] == '>') {
            continue;
        }
        for (char c : line) {
            if (!isspace(static_cast<unsigned char>(c))) {
                sequence += c;
            }
        }
    }
    return sequence;
}


int processUserInput(int argc, char* argv[]) {

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
    int match_score = DEFAULT_MATCH_SCORE;
    int mismatch_penalty = DEFAULT_MISMATCH_PENALTY;
    int gap_penalty = DEFAULT_GAP_PENALTY;

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

    std::cout << seq1.length() << std::endl;
    std::cout << "--------------------" << std::endl;
    std::cout << seq2.length() << std::endl;

    LocalAlignmentFinder laf (
        seq1,
        seq2,
        match_score,
        mismatch_penalty,
        gap_penalty
    );


    std::cout << laf << std::endl;

    return 0;
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        //std::string s1 = "ATGCTGACCGTACGTAGCTAGCTGACTGATCGTGACTGACGTCGATGCTAGCTAGCTGATCGATCGTACGATGCTAGCTAGCATCGATCGTGATCGTAGCTGACGTAGCTAGCTAGCAGTCTAGCTAGCTAGTGCTAGCTAGTCGATCGTAGCTAGTCGATCGTAGCTAGCTAGCTAGGCTAGCTAGTGCTAGCTAGGCTAGCTAGTCGATCATGCTGACCGTACGTAGCTAGCTGACTGATCGTGACTGACGTCGATGCTAGCTAGCTGATCGATCGTACGATGCTAGCTAGCATCGATCGTGATCGTAGCTGACGTAGCTAGCTAGCAGTCTAGCTAGCTAGTGCTAGCTAGTCGATCGTAGCTAGTCGATCGTAGCTAGCTAGCTAGGCTAGCTAGTGCTAGCTAGGCTAGCTAGTCGATCATGCTGACCGTACGTAGCTAGCTGACTGATCGTGACTGACGTCGATGCTAGCTAGCTGATCGATCGTACGATGCTAGCTAGCATCGATCGTGATCGTAGCTGACGTAGCTAGCTAGCAGTCTAGCTAGCTAGTGCTAGCTAGTCGATCGTAGCTAGTCGATCGTAGCTAGCTAGCTAGGCTAGCTAGTGCTAGCTAGGCTAGCTAGTCGATCATGCTGACCGTACGTAGCTAGCTGACTGATCGTGACTGACGTCGATGCTAGCTAGCTGATCGATCGTACGATGCTAGCTAGCATCGATCGTGATCGTAGCTGACGTAGCTAGCTAGCAGTCTAGCTAGCTAGTGCTAGCTAGTCGATCGTAGCTAGTCGATCGTAGCTAGCTAGCTAGGCTAGCTAGTGCTAGCTAGGCTAGCTAGTCGATCATGCTGACCGTACGTAGCTAGCTGACTGATCGTGACTGACGTCGATGCTAGCTAGCTGATCGATCGTACGATGCTAGCTAGCATCGATCGTGATCGTAGCTGACGTAGCTAGCTAGCAGTCTAGCTAGCTAGTGCTAGCTAGTCGATCGTAGCTAGTCGATCGTAGCTAGCTAGCTAGGCTAGCTAGTGCTAGCTAGGCTAGCTAGTCGATCATGCTGACCGTACGTAGCTAGCTGACTGATCGTGACTGACGTCGATGCTAGCTAGCTGATCGATCGTACGATGCTAGCTAGCATCGATCGTGATCGTAGCTGACGTAGCTAGCTAGCAGTCTAGCTAGCTAGTGCTAGCTAGTCGATCGTAGCTAGTCGATCGTAGCTAGCTAGCTAGGCTAGCTAGTGCTAGCTAGGCTAGCTAGTCGATCATGCTGACCGTACGTAGCTAGCTGACTGATCGTGACTGACGTCGATGCTAGCTAGCTGATCGATCGTACGATGCTAGCTAGCATCGATCGTGATCGTAGCTGACGTAGCTAGCTAGCAGTCTAGCTAGCTAGTGCTAGCTAGTCGATCGTAGCTAGTCGATCGTAGCTAGCTAGCTAGGCTAGCTAGTGCTAGCTAGGCTAGCTAGTCGATCATGCTGACCGTACGTAGCTAGCTGACTGATCGTGACTGACGTCGATGCTAGCTAGCTGATCGATCGTACGATGCTAGCTAGCATCGATCGTGATCGTAGCTGACGTAGCTAGCTAGCAGTCTAGCTAGCTAGTGCTAGCTAGTCGATCGTAGCTAGTCGATCGTAGCTAGCTAGCTAGGCTAGCTAGTGCTAGCTAGGCTAGCTAGTCGATCATGCTGACCGTACGTAGCTAGCTGACTGATCGTGACTGACGTCGATGCTAGCTAGCTGATCGATCGTACGATGCTAGCTAGCATCGATCGTGATCGTAGCTGACGTAGCTAGCTAGCAGTCTAGCTAGCTAGTGCTAGCTAGTCGATCGTAGCTAGTCGATCGTAGCTAGCTAGCTAGGCTAGCTAGTGCTAGCTAGGCTAGCTAGTCGATCATGCTGACCGTACGTAGCTAGCTGACTGATCGTGACTGACGTCGATGCTAGCTAGCTGATCGATCGTACGATGCTAGCTAGCATCGATCGTGATCGTAGCTGACGTAGCTAGCTAGCAGTCTAGCTAGCTAGTGCTAGCTAGTCGATCGTAGCTAGTCGATCGTAGCTAGCTAGCTAGGCTAGCTAGTGCTAGCTAGGCTAGCTAGTCGATCATGCTGACCGTACGTAGCTAGCTGACTGATCGTGACTGACGTCGATGCTAGCTAGCTGATCGATCGTACGATGCTAGCTAGCATCGATCGTGATCGTAGCTGACGTAGCTAGCTAGCAGTCTAGCTAGCTAGTGCTAGCTAGTCGATCGTAGCTAGTCGATCGTAGCTAGCTAGCTAGGCTAGCTAGTGCTAGCTAGGCTAGCTAGTCGATCATGCTGACCGTACGTAGCTAGCTGACTGAT";
        //std::string s2 = "GGCTTTTTTTTAGCTAA";

        const std::string s1 = "GTCTAC";
        const std::string s2 = "TCTCGAT";

        LocalAlignmentFinder laf (
            s1.c_str(),
            s2.c_str(),
            DEFAULT_MATCH_SCORE,
            DEFAULT_MISMATCH_PENALTY,
            DEFAULT_GAP_PENALTY
        );


        // laf.print_matrix();

        const std::string expected_str = laf.toString();

        std::cout << expected_str << std::endl;
        std::cout << laf << std::endl;

        return 0;
    }
    else {
        auto start = std::chrono::high_resolution_clock::now();
        processUserInput(argc, argv);
        auto end = std::chrono::high_resolution_clock::now();
        // Calculate duration
        std::chrono::duration<double> duration = end - start;

        // Output duration in seconds
        std::cout << "Duration: " << duration.count() << " seconds" << std::endl;

        return 0;
    }
}


/*
 ******************************************************************************
 *
 *  score_matrix.h
 *
 *  Copyright (c) 2025 John Pazarzis
 *  Licensed under the GPL License.
 *
 *  This header defines the ScoreMatrix class, which implements the core logic
 *  for the Smith-Waterman local alignment algorithm. The ScoreMatrix manages
 *  the dynamic programming matrix, scoring configuration, and traceback
 *  procedures necessary to compute optimal local alignments between two
 *  input sequences.
 *
 *  Usage:
 *      - Construct a ScoreMatrix object by providing two sequences and
 *        scoring parameters (match score, mismatch penalty, gap penalty).
 *      - Call getLocalAlignment() to obtain the highest-scoring local
 *        alignments as a vector of strings.
 *      - Additional methods provide matrix details, such as the
 *        optimal score position and the alignment matrix as a string.
 *
 *  The ScoreMatrix class disables copy and move operations to prevent
 *  unintended copying of potentially large alignment data.
 *
 ******************************************************************************
 */

#ifndef SCORE_MATRIX_INCLUDED
#define SCORE_MATRIX_INCLUDED

#include <string>
#include <utility>
#include <vector>
#include <ostream>

class ScoreMatrix {

public:
    ScoreMatrix(const std::string& s1, const std::string& s2,
            int match_score, int mismatch_penalty, int gap_penalty, size_t max_alignments = 10);
    ~ScoreMatrix();

    // Disable copy and move operations
    ScoreMatrix() = delete;
    ScoreMatrix(const ScoreMatrix&) = delete;
    ScoreMatrix& operator=(const ScoreMatrix&) = delete;
    ScoreMatrix(ScoreMatrix&&) = delete;
    ScoreMatrix& operator=(ScoreMatrix&&) = delete;

    // Accessors.
    const std::string& getSequence1() const;
    const std::string& getSequence2() const;
    int getRowCount() const;
    int getColCount() const;
    int getScore(int row, int col) const; // throws std::out_of_range.
    int getMaxScore() const;
    std::string toString() const;
    const std::vector<std::string>& getLocalAlignments() const;
    std::string getLocalAlignmentsAsJson() const;
    size_t getNumberOfAlignments() const;

private:
    // Private functions.
    void initializeMatrix();
    void traceback(int row, int col, std::string x1, std::string x2, std::string a);
    void processDiagonal(int col, int starting_row);

private:
    // Passed from the user.
    const std::string m_sequence1;
    const std::string m_sequence2;
    const int m_match_score;
    const int m_mismatch_penalty;
    const int m_gap_penalty;
    const size_t m_max_alignments;

    // Keep internal state.
    std::vector<std::pair<int, int>> m_max_positions;
    std::vector<std::string> m_local_alignments;
    int m_max_score;

    // m_matrix is a flat matrix allocated as a contiguous memory block
    // for compatibility with CUDA device memory transfers.
    // Do NOT use std::vector or smart pointers here.
    int* m_matrix;

};

std::ostream& operator<<(std::ostream& os, const ScoreMatrix& obj);

#endif // SCORE_MATRIX_INCLUDED

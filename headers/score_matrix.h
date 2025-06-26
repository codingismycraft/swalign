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

#ifndef __SCORE_MATRIX_H__
#define __SCORE_MATRIX_H__

#include <string>
#include <utility>
#include <vector>


class ScoreMatrix {

private:
    const std::string m_sequence1;
    const std::string m_sequence2;
    const int m_match_score;
    const int m_mismatch_penalty;
    const int m_gap_penalty;

    mutable std::vector<std::pair<int, int>> m_max_positions;
    mutable std::pair<int, int> m_max_position;
    mutable std::vector<std::vector<int>> m_matrix;
    mutable std::vector<std::string> m_local_alignments;

    void initializeMatrix() const;
    void traceback(int row, int col, std::string x1, std::string x2, std::string a) const;
public:
    ScoreMatrix(const std::string& s1, const std::string& s2,
            int match_score, int mismatch_penalty, int gap_penalty);
    ~ScoreMatrix() = default;

    // Disable copy and move operations
    ScoreMatrix(const ScoreMatrix&) = delete;
    ScoreMatrix& operator=(const ScoreMatrix&) = delete;
    ScoreMatrix(ScoreMatrix&&) = delete;
    ScoreMatrix& operator=(ScoreMatrix&&) = delete;

    std::string getSequence1() const;
    std::string getSequence2() const;
    std::pair<int, int> getMaxPosition() const;
    std::vector<std::string> getLocalAlignment() const;
    int getRowCount() const;
    int getColCount() const;
    int getScore(int row, int col) const;
    std::string to_str() const;
};


#endif // __SCORE_MATRIX_H__

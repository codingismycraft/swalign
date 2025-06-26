/*
 ******************************************************************************
 *
 *  score_matrix.cpp
 *
 *  Copyright (c) 2025 John Pazarzis
 *  Licensed under the GPL License.
 *
 *  This file implements the ScoreMatrix class, which provides the core logic
 *  for the Smith-Waterman local alignment algorithm. This includes:
 *    - Initializing and filling the dynamic programming matrix
 *    - Performing traceback to reconstruct all optimal local alignments
 *    - Accessor methods for sequences, matrix scores, and alignments
 *    - Generating a string representation of the alignment matrix
 *
 *  The ScoreMatrix class is intended to be used as part of the swalign
 *  program for efficient and robust local sequence alignment of nucleotide
 *  or protein sequences.
 *
 ******************************************************************************
 */

#include "score_matrix.h"

#include <algorithm>
#include <assert.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>


// Implementation of ScoreMatrix class.
ScoreMatrix::ScoreMatrix( const std::string& s1, const std::string& s2,
        int match_score, int mismatch_penalty, int gap_penalty):
            m_sequence1(s1),
            m_sequence2(s2),
            m_match_score(match_score),
            m_mismatch_penalty(mismatch_penalty),
            m_gap_penalty(gap_penalty)

{
    initializeMatrix();
}

std::vector<std::string> ScoreMatrix::getLocalAlignment() const
{
    return m_local_alignments;
}

std::string ScoreMatrix::getSequence1() const {
    return m_sequence1;
}

std::string ScoreMatrix::getSequence2() const {
    return m_sequence2;
}


int ScoreMatrix::getRowCount() const {
    return m_sequence2.length() + 1;
}

int ScoreMatrix::getColCount() const {
    return m_sequence1.length() + 1;
}

int ScoreMatrix::getScore(int row, int col) const {
    return m_matrix[row][col];
}


void ScoreMatrix::initializeMatrix() const{
    m_local_alignments.clear();
    m_max_positions.clear();
    m_matrix.push_back(std::vector<int>(getColCount(), 0));

    m_max_position.first = 0;
    m_max_position.second = 0;

    int max_score = 0;

    for (int i = 1; i < getRowCount(); ++i) {
        std::vector<int> row;
        row.push_back(0);
        for(int j = 1; j < getColCount(); ++j) {
            int score_diagonal = 0;

            if (m_sequence1[j - 1] == m_sequence2[i - 1])
            {
                score_diagonal = m_matrix[i - 1][j - 1] + m_match_score;
            }
            else
            {
                score_diagonal = m_matrix[i - 1][j - 1] + m_mismatch_penalty;
            }


            const int score_up = m_matrix[i - 1][j] + m_gap_penalty;
            const int score_left = row[j - 1] + m_gap_penalty;
            const int score = std::max({score_diagonal, score_up, score_left, 0});

            if (score > max_score) {
                m_max_positions.clear(); // Clear previous max positions if we
            }


            if (score >= max_score) {
                max_score = score;
                m_max_position.first = i;
                m_max_position.second = j;

                m_max_positions.push_back(std::make_pair(i, j));
            }

            row.push_back(score);
        }
        m_matrix.push_back(row);
    }

    for (const auto& pos : m_max_positions) {
        traceback(pos.first, pos.second, "", "", "");
    }
}

std::pair<int, int> ScoreMatrix::getMaxPosition() const {
    return m_max_position;
}

std::string ScoreMatrix::to_str() const {
    std::ostringstream oss;
    const int width = 6;
    if (m_matrix.empty()) return "";

    // Build horizontal separator
    std::string separator;
    // The separator needs to cover one extra for the row label column.
    for (size_t col = 0; col < m_matrix[0].size() + 1; ++col) {
        separator += std::string(width, '_');
        if (col + 1 < m_matrix[0].size() + 1) separator += "|";
    }
    separator += "\n";

    // Print top separator
    oss << separator;

    // Print header row: empty cell, then m_sequence1 characters as column headers
    oss << std::setw(width) << ' ';

    oss << "|" << std::setw(width) << ' ';
    for (size_t col = 0; col < m_sequence1.size(); ++col) {
        oss << "|" << std::setw(width) << m_sequence1[col];
    }
    // The number of columns for the sequence is m_sequence1.size(), which matches m_matrix[0].size()-1,
    // so we need one more empty cell at the end to match the bottom rows.
    oss << "\n" << separator;

    // Print all matrix rows
    for (size_t row = 0; row < m_matrix.size(); ++row) {
        // First column: empty for first row, else m_sequence2 character
        if (row == 0) {
            oss << std::setw(width) << ' ';
        } else {
            oss << std::setw(width) << m_sequence2[row - 1];
        }
        // Print all matrix columns for this row
        for (size_t col = 0; col < m_matrix[row].size(); ++col) {
            oss << "|" << std::setw(width) << m_matrix[row][col];
        }
        oss << "\n" << separator;
    }

    return oss.str();
}

void ScoreMatrix::traceback(int row, int col, std::string x1, std::string x2, std::string a) const {
    while(row >=0 && col >=0 && m_matrix[row][col] > 0) {
        const int row_coming_in = row;
        const int col_coming_in = col;

        const std::string a_coming_in = a;
        const std::string x1_coming_in = x1;
        const std::string x2_coming_in = x2;

        const int current_score = m_matrix[row][col];
        const int diagonal_row = row - 1;
        const int diagonal_col = col - 1;

        const bool valid_diagonal = (diagonal_row >= 0 && diagonal_col >= 0);
        const bool valid_up_col = (col - 1 >= 0);
        const bool valid_left_row = (row - 1 >= 0);

        bool already_moved = false;


        if (valid_diagonal && m_matrix[diagonal_row][diagonal_col] + m_match_score == current_score) {
            if (!already_moved) {
                a = '*' + a_coming_in;
                x1 = m_sequence1[col_coming_in - 1] + x1_coming_in;
                x2 = m_sequence2[row_coming_in  - 1] + x2_coming_in;
                row = diagonal_row;
                col = diagonal_col;
                already_moved = true;
            }
            else {
                assert (false && "Traceback logic error: already_moved should not be true here");
            }
        }

        if (valid_diagonal && m_matrix[diagonal_row][diagonal_col] + m_mismatch_penalty == current_score) {
            if (!already_moved) {
                a = '|' + a_coming_in;
                x1 = m_sequence1[col_coming_in - 1] + x1_coming_in;
                x2 = m_sequence2[row_coming_in  - 1] + x2_coming_in;
                row = diagonal_row;
                col = diagonal_col;
                already_moved = true;
            }
            else {
                traceback(
                    diagonal_row,
                    diagonal_col,
                    m_sequence1[col_coming_in - 1] + x1_coming_in,
                    m_sequence2[row_coming_in  - 1] + x2_coming_in,
                    '|' + a_coming_in
               );
            }
        }

        if (valid_left_row && m_matrix[row_coming_in  - 1][col_coming_in ] + m_gap_penalty == current_score) {
            if (!already_moved) {
                a = ' ' + a;
                x1 = '_' + x1;
                x2= m_sequence2[row_coming_in  - 1] + x2;
                row -= 1;
                already_moved = true;
            }
            else {
                traceback(
                    row_coming_in  - 1,
                    col_coming_in,
                    '_' + x1_coming_in,
                    m_sequence2[row_coming_in  - 1] + x2_coming_in,
                    ' ' + a_coming_in
               );
            }
        }

        if (valid_up_col && m_matrix[row_coming_in ][col_coming_in  - 1] + m_gap_penalty == current_score) {
            if (!already_moved) {
                a = ' ' + a;
                x1 = m_sequence1[col - 1] + x1;
                x2 = '_' + x2;
                col  -= 1;
                already_moved = true;
            }
            else {
                traceback(
                    row_coming_in,
                    col_coming_in  - 1,
                    m_sequence1[col_coming_in - 1] + x1_coming_in,
                    '_' + x2_coming_in,
                    ' ' + a_coming_in
               );
            }
        }

       assert(already_moved && "Traceback logic error");
       assert (row < row_coming_in || col < col_coming_in);
    }

    m_local_alignments.push_back("\n"  + x2 + "\n" + a + "\n" + x1 + "\n");
}


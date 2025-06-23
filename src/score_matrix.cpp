// Implementation of ScoreMatrix class and related details.

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
    if (m_sequence2.length() > m_sequence1.length()) {
        m_sequence1 = s2;
        m_sequence2 = s1;
    }

    initializeMatrix();
}

std::string ScoreMatrix::getLocalAlignment() const
{
    return m_local_alignment;
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

            if (score >= max_score) {
                max_score = score;
                m_max_position.first = i;
                m_max_position.second = j;
            }

            row.push_back(score);
        }
        m_matrix.push_back(row);
    }

    traceback();
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

void ScoreMatrix::traceback() const {
    int row = m_max_position.first;
    int col = m_max_position.second;

    std::string x1, x2, a;

    while( row >=0 && col >=0 && m_matrix[row][col] > 0) {
        const int current_score = m_matrix[row][col];
        const int diagonal_row = row - 1;
        const int diagonal_col = col - 1;

        if (m_matrix[diagonal_row][diagonal_col] + m_match_score == current_score) {
            a = '*' + a;
            x1 = m_sequence1[col - 1] + x1;
            x2 = m_sequence2[row - 1] + x2;
            row = diagonal_row;
            col = diagonal_col;
        } else if (m_matrix[diagonal_row][diagonal_col] + m_mismatch_penalty == current_score) {
            a = '|' + a;
            x1 = m_sequence1[col - 1] + x1;
            x2 = m_sequence2[row - 1] + x2;
            row = diagonal_row;
            col = diagonal_col;
        } else if (m_matrix[row - 1][col] + m_gap_penalty == current_score) {
            a = ' ' + a;
            x2= m_sequence1[row - 1] + x2;
            x1 = '_' + x1;
            row -= 1;
        } else if (m_matrix[row][col - 1] + m_gap_penalty == current_score) {
            a = ' ' + a;
            x1 = m_sequence1[col - 1] + x1;
            x2 = '_' + x2;
            col -= 1;
        } else {
            assert(false && "Traceback logic error");
        }
    }

    m_local_alignment =  "\n"  + x1 + "\n" + a + "\n" + x2 + "\n";
}


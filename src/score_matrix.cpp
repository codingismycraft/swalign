// Implementation of ScoreMatrix class and related details.

#include "score_matrix.h"

#include <algorithm>
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


            int score_up = m_matrix[i - 1][j] + m_gap_penalty;
            int score_left = row[j - 1] + m_gap_penalty;

            row.push_back(std::max({score_diagonal, score_up, score_left, 0}));
        }
        m_matrix.push_back(row);
    }
}

std::string ScoreMatrix::to_str() const {
    std::ostringstream oss;
    const int width = 6;
    if (m_matrix.empty()) return "";

    // Build horizontal separator
    std::string separator;
    for (size_t col = 0; col < m_matrix[0].size(); ++col) {
        separator += std::string(width, '_');
        if (col + 1 < m_matrix[0].size()) separator += "|";
    }
    separator += "\n";

    // Header separator
    oss << separator;
    // Rows
    for (const auto& row : m_matrix) {
        for (size_t col = 0; col < row.size(); ++col) {
            oss << std::setw(width) << row[col];
            if (col + 1 < row.size()) oss << "|";
        }
        oss << "\n" << separator;
    }
    return oss.str();
}


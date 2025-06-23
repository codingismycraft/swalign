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


            int score_up = m_matrix[i - 1][j] + m_gap_penalty;
            int score_left = row[j - 1] + m_gap_penalty;

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


/*
Traceback Mechanism in Smith-Waterman Algorithm

After filling the dynamic programming matrix (H) according to the
Smith-Waterman algorithm for local sequence alignment, the traceback step is
used to recover the optimal local alignment. The mechanism works as follows:

1. Identify the cell in H with the highest score. This cell marks the end of
the best local alignment.

2. Begin the traceback from this highest-scoring cell. At each step:

- If the current cell’s score came from a diagonal move (match/mismatch), move
diagonally (i-1, j-1).

- If the current cell’s score came from an up move (gap in sequence B), move up
(i-1, j).

- If the current cell’s score came from a left move (gap in sequence A), move
left (i, j-1).

- Stop the traceback when a cell with score 0 is reached (the beginning of the
local alignment).

3. As you traceback, record each move:
   - Diagonal: match or mismatch
   - Up: gap in sequence B
   - Left: gap in sequence A

4. After reaching a cell with score 0, reverse the recorded moves to construct
the optimal local alignment.

In summary, the traceback in Smith-Waterman starts from the highest-scoring
cell and follows the path of score origins (diagonal, up, or left) until a cell
with score zero is reached, producing the best local alignment.
*/
void ScoreMatrix::traceback() const {
    std::vector<char> longest_sequence;
    m_local_alignment = "";
    int row = m_max_position.first;
    int col = m_max_position.second;

    const char gap = '-';

    for(;;){
        if (row <= 0 || col <=0)
        {
            break;
        }

        const int current_score = m_matrix[row][col];

        if (current_score <= 0 )
        {
            break;
        }

        const bool matching_chars = m_sequence2[row-1] == m_sequence1[col - 1];

        if (matching_chars)
        {
            longest_sequence.push_back(m_sequence1[col - 1]);
        }
        else
        {
            longest_sequence.push_back(gap);
        }


        const int diagonal_row = row - 1;
        const int diagonal_col = col - 1;

        if (m_matrix[diagonal_row][diagonal_col] + m_match_score == current_score)
        {
            row = diagonal_row;
            col = diagonal_col;
            continue;
        }
        else if (m_matrix[diagonal_row][diagonal_col] + m_mismatch_penalty == current_score)
        {
            row = diagonal_row;
            col = diagonal_col;
            continue;
        }
        else if (m_matrix[row - 1][col] + m_gap_penalty == current_score)
        {
            row -= 1;
            continue;
        }
        else if (m_matrix[row][col - 1] + m_gap_penalty == current_score)
        {
            col -= 1;
            continue;
        }

        assert(false && "Traceback logic error");
    }

    std::reverse(longest_sequence.begin(), longest_sequence.end());
    m_local_alignment = std::string(longest_sequence.begin(), longest_sequence.end());
}


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

#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include "rapidjson/document.h"

#include <algorithm>
#include <assert.h>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

    inline int getMatrixValue(const int row, const int col, const int col_count, const int* matrix) {
        return matrix[row * col_count + col];
    }

    inline void setMatrixValue(const int value, const int row, const int col, const int col_count, int* matrix) {
        matrix[row * col_count + col] = value;
    }
}



// Implementation of ScoreMatrix class.
ScoreMatrix::ScoreMatrix( const std::string& s1, const std::string& s2,
        int match_score, int mismatch_penalty, int gap_penalty, size_t max_alignments):
            m_sequence1(s1),
            m_sequence2(s2),
            m_match_score(match_score),
            m_mismatch_penalty(mismatch_penalty),
            m_gap_penalty(gap_penalty),
            m_max_alignments(max_alignments),
            m_max_score(0)

{
    m_matrix = new int[(s1.length() + 1) * (s2.length() + 1)];
    initializeMatrix();
}

ScoreMatrix::~ScoreMatrix() {
    if (m_matrix) {
        delete[] m_matrix;
        m_matrix = nullptr;
    }
}


const std::vector<std::string>& ScoreMatrix::getLocalAlignments() const {
    return m_local_alignments;
}

const std::string& ScoreMatrix::getSequence1() const {
    return m_sequence1;
}

const std::string& ScoreMatrix::getSequence2() const {
    return m_sequence2;
}


int ScoreMatrix::getRowCount() const {
    return m_sequence2.length() + 1;
}

int ScoreMatrix::getColCount() const {
    return m_sequence1.length() + 1;
}

int ScoreMatrix::getScore(int row, int col) const {
    if (row < 0 || row >= getRowCount() || col < 0 || col >= getColCount()) {
        throw std::out_of_range("Row or column index out of bounds");
    }
    return getMatrixValue(row, col, getColCount(), m_matrix);
}

size_t ScoreMatrix::getNumberOfAlignments() const {
    return m_local_alignments.size();
}

int ScoreMatrix::getMaxScore() const {
    return m_max_score;
}

void ScoreMatrix::initializeMatrix() {
    m_local_alignments.clear();
    m_max_positions.clear();
    m_max_score = 0;

    for(int col = 1; col < getColCount(); ++col) {
        processDiagonal(col, 1);
    }

    const int last_col = getColCount() - 1;

    for (int row = 1; row < getRowCount(); ++row) {
        processDiagonal(last_col, row);
    }

    findMaxScores();

    for (const auto& pos : m_max_positions) {
        traceback(pos.first, pos.second, "", "", "");
    }
}

void ScoreMatrix::findMaxScores() {
    int current_max = -1;
    m_max_positions.clear();

    const int cols = getColCount();
    const int rows = getRowCount();

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            const int value = m_matrix[row * cols + col];
            if (value > current_max) {
                m_max_positions.clear();
            }
            if (value >= current_max) {
                current_max = value;
                m_max_positions.push_back(std::make_pair(row, col));
            }
        }
    }

    m_max_score = current_max;
}


void ScoreMatrix::processDiagonal(int col, int starting_row){
    assert (col >= 0 && col < getColCount() && "Column index out of bounds");
    for (int row = starting_row; col > 0 && row < getRowCount(); ++row) {

        const int& diagonal_value = getMatrixValue(row-1, col-1, getColCount(), m_matrix);
        const int& up_value = getMatrixValue(row - 1, col, getColCount(), m_matrix);
        const int& left_value = getMatrixValue(row, col - 1, getColCount(), m_matrix);

        const int score_diagonal = diagonal_value + (m_sequence1[col - 1] == m_sequence2[row - 1] ? m_match_score : m_mismatch_penalty);
        const int score_up = up_value + m_gap_penalty;
        const int score_left = left_value + m_gap_penalty;

        const int score = std::max({score_diagonal, score_up, score_left, 0});

        setMatrixValue(score, row, col, getColCount(), m_matrix);

        --col;
    }
}

std::string ScoreMatrix::toString() const {
    std::ostringstream oss;
    const int width = 6;
    if (m_matrix == nullptr) return "";

    // Build horizontal separator
    std::string separator;
    // The separator needs to cover one extra for the row label column.
    for (int col = 0; col < getColCount() + 1; ++col) {
        separator += std::string(width, '_');
        if (col + 1 < getColCount() + 1) separator += "|";
    }
    separator += "\n";

    // Print top separator
    oss << separator;

    // Print header row: empty cell, then m_sequence1 characters as column headers
    oss << std::setw(width) << ' ';

    oss << "|" << std::setw(width) << ' ';
    for (int col = 0; col < getColCount()-1; ++col) {
        oss << "|" << std::setw(width) << m_sequence1[col];
    }
    oss << "\n" << separator;

    // Print all matrix rows
    for (int row = 0; row < getRowCount(); ++row) {
        // First column: empty for first row, else m_sequence2 character
        if (row == 0) {
            oss << std::setw(width) << ' ';
        } else {
            oss << std::setw(width) << m_sequence2[row - 1];
        }
        // Print all matrix columns for this row
        for (int col = 0; col < getColCount() ; ++col) {
            oss << "|" << std::setw(width) << getMatrixValue(row, col, getColCount(), m_matrix);
        }
        oss << "\n" << separator;
    }

    return oss.str();
}

void ScoreMatrix::traceback(int row, int col, std::string x1, std::string x2, std::string a) {

    while(row >=0 && col >=0 && getScore(row, col) > 0) {
        if (m_local_alignments.size() >= m_max_alignments)
        {
            return;
        }

        const int row_coming_in = row;
        const int col_coming_in = col;

        const std::string a_coming_in = a;
        const std::string x1_coming_in = x1;
        const std::string x2_coming_in = x2;

        const int current_score = getScore(row, col);
        const int diagonal_row = row - 1;
        const int diagonal_col = col - 1;

        const bool valid_diagonal = (diagonal_row >= 0 && diagonal_col >= 0);
        const bool valid_up_col = (col - 1 >= 0);
        const bool valid_left_row = (row - 1 >= 0);

        bool already_moved = false;

        const bool same_char = (m_sequence1[col-1] == m_sequence2[row-1]);

        if (same_char && valid_diagonal && getScore(diagonal_row, diagonal_col) + m_match_score == current_score) {
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

        if (!same_char && valid_diagonal && getScore(diagonal_row, diagonal_col) + m_mismatch_penalty == current_score) {
            if (!already_moved) {
                a = '|' + a_coming_in;
                x1 = m_sequence1[col_coming_in - 1] + x1_coming_in;
                x2 = m_sequence2[row_coming_in  - 1] + x2_coming_in;
                row = diagonal_row;
                col = diagonal_col;
                already_moved = true;
            }
            else {
                if (m_local_alignments.size() < m_max_alignments) {
                    traceback(
                        diagonal_row,
                        diagonal_col,
                        m_sequence1[col_coming_in - 1] + x1_coming_in,
                        m_sequence2[row_coming_in  - 1] + x2_coming_in,
                        '|' + a_coming_in
                   );
               }
            }
        }


        if (valid_left_row && getScore(row_coming_in  - 1, col_coming_in) + m_gap_penalty == current_score) {
            if (!already_moved) {
                a = ' ' + a;
                x1 = '_' + x1;
                x2= m_sequence2[row_coming_in  - 1] + x2;
                row -= 1;
                already_moved = true;
            }
            else {
                if (m_local_alignments.size() < m_max_alignments) {
                    traceback(
                        row_coming_in  - 1,
                        col_coming_in,
                        '_' + x1_coming_in,
                        m_sequence2[row_coming_in  - 1] + x2_coming_in,
                        ' ' + a_coming_in
                   );
                }
            }
        }

        if (valid_up_col && getScore(row_coming_in, col_coming_in  - 1) + m_gap_penalty == current_score) {
            if (!already_moved) {
                a = ' ' + a;
                x1 = m_sequence1[col - 1] + x1;
                x2 = '_' + x2;
                col  -= 1;
                already_moved = true;
            }
            else {
                if (m_local_alignments.size() < m_max_alignments) {
                    traceback(
                        row_coming_in,
                        col_coming_in  - 1,
                        m_sequence1[col_coming_in - 1] + x1_coming_in,
                        '_' + x2_coming_in,
                        ' ' + a_coming_in
                   );
                }
            }
        }

       assert(already_moved && "Traceback logic error");
       assert (row < row_coming_in || col < col_coming_in);
    }
    a = a + " " + std::to_string(evaluateScore(a));
    m_local_alignments.push_back("\n"  + x2 + "\n" + a + "\n" + x1 + "\n");
}

int32_t ScoreMatrix::evaluateScore(const std::string& alighmentStr) const {
    int starCount = 0;
    int pipeCount = 0;
    int spaceCount = 0;

    for (char ch : alighmentStr) {
        if (ch == '*') {
            ++starCount;
        } else if (ch == '|') {
            ++pipeCount;
        } else if (ch == ' ') {
            ++spaceCount;
        }
    }

    return starCount * m_match_score +
           pipeCount * m_mismatch_penalty +
           spaceCount * m_gap_penalty;
}


// Assumes input is groups of 3 non-empty lines separated by empty lines
std::string ScoreMatrix::getLocalAlignmentsAsJson() const {

    const std::string input = std::accumulate(
            m_local_alignments.begin(), m_local_alignments.end(), std::string()
    );

    std::istringstream ss(input);
    std::string line;
    std::vector<std::tuple<std::string, std::string, std::string>> entries;

    while (std::getline(ss, line)) {
        // skip empty lines
        if (line.empty())
            continue;
        std::string seq1 = line;
        std::string match, seq2;
        if (!std::getline(ss, match)) break;
        if (!std::getline(ss, seq2)) break;
        entries.emplace_back(seq1, match, seq2);
    }

    rapidjson::Document d;
    d.SetArray();
    auto& allocator = d.GetAllocator();

    for (const auto& entry : entries) {
        rapidjson::Value obj(rapidjson::kObjectType);
        obj.AddMember("seq1", rapidjson::Value(std::get<0>(entry).c_str(), allocator), allocator);
        obj.AddMember("match", rapidjson::Value(std::get<1>(entry).c_str(), allocator), allocator);
        obj.AddMember("seq2", rapidjson::Value(std::get<2>(entry).c_str(), allocator), allocator);
        d.PushBack(obj, allocator);
    }

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    d.Accept(writer);

    return buffer.GetString();
}

std::ostream& operator<<(std::ostream& os, const ScoreMatrix& obj) {
    os << "\n***************************************" << std::endl;
    os << "Seq1: " << obj.getSequence1() << std::endl;
    os << "Seq2: " << obj.getSequence2() << std::endl;


    int counter = 1;
    os << "\n--------------------------" << std::endl;
    for (const auto& aln : obj.getLocalAlignments()) {
         os << "Alignment num: " << counter++ << "\n";
         os << aln << std::endl;
         os << "--------------------------" << std::endl;
    }

    os << "\nNumber of Alignments ..: " << obj.getNumberOfAlignments()<< std::endl;
    os << "Max score .............: " << obj.getMaxScore()<< std::endl;
    return os;
}


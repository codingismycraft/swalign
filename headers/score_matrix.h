#ifndef __SCORE_MATRIX_H__
#define __SCORE_MATRIX_H__

#include <string>
#include <utility> // std::pair
#include <vector>


class ScoreMatrix {

private:
    const std::string m_sequence1;
    const std::string m_sequence2;
    const int m_match_score;
    const int m_mismatch_penalty;
    const int m_gap_penalty;
    mutable std::pair<int, int> m_max_position;
    mutable std::vector<std::vector<int>> m_matrix;
    mutable std::vector<std::string> m_local_alignments;

    void initializeMatrix() const;
    void traceback() const;
public:
    ScoreMatrix(const std::string& s1, const std::string& s2,
            int match_score, int mismatch_penalty, int gap_penalty);
    ~ScoreMatrix() = default;

    // Disable copy and move operations
    ScoreMatrix(const ScoreMatrix&) = delete;            // Copy constructor
    ScoreMatrix& operator=(const ScoreMatrix&) = delete; // Copy assignment
    ScoreMatrix(ScoreMatrix&&) = delete;                 // Move constructor
    ScoreMatrix& operator=(ScoreMatrix&&) = delete;      // Move assignment

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

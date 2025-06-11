#ifndef __SCORE_MATRIX_H__
#define __SCORE_MATRIX_H__

#include <string>
#include <vector>


class ScoreMatrix {

private:
    const std::string m_sequence1;
    const std::string m_sequence2;
    const int m_match_score;
    const int m_mismatch_penalty;
    const int m_gap_penalty;

    mutable std::vector<std::vector<int>> m_matrix;

    void initializeMatrix() const;
public:
    ScoreMatrix(const std::string& s1, const std::string& s2,
            int match_score, int mismatch_penalty, int gap_penalty);
    virtual ~ScoreMatrix() = default;

    std::string getSequence1() const;
    std::string getSequence2() const;
    int getRowCount() const;
    int getColCount() const;
    int getScore(int row, int col) const;
    std::string to_str() const;
};





#endif // __SCORE_MATRIX_H__

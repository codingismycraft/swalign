#include "score_matrix.h"

#include <iostream>
#include <string>
#include <assert.h>


void testScoreMatrix(){
    const std::string s1 = "ACGTC";
    const std::string s2 = "ACATC";

    const int expected_scores[6][6] = {
        {0, 0, 0, 0, 0, 0},
        {0, 2, 0, 0, 0, 0},
        {0, 0, 4, 2, 0, 2},
        {0, 2, 2, 3, 1, 0},
        {0, 0, 1, 1, 5, 3},
        {0, 0, 2, 0, 3, 7}
    };


    ScoreMatrix scoreMatrix(s1, s2, 2, -1, -2);

    const int expected_row_count = s2.length() + 1;
    const int expected_col_count = s1.length() + 1;

    assert(scoreMatrix.getSequence1() == s1);
    assert(scoreMatrix.getSequence2() == s2);

    assert(scoreMatrix.getRowCount() == expected_row_count);
    assert(scoreMatrix.getColCount() == expected_col_count);

    const std::string expected_str = scoreMatrix.to_str();

    std::cout << expected_str << std::endl;


    for(int row =0; row < 6; ++row) {
        for(int col = 0; col < 6; ++col) {
            const int retrieved = scoreMatrix.getScore(row, col);
            const int expected = expected_scores[row][col];
            assert(retrieved == expected);
        }
    }
}




int main(){
    testScoreMatrix();
}



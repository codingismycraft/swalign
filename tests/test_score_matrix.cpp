/*
 ******************************************************************************
 *
 *  test_score_matrix.cpp
 *
 *  Copyright (c) 2025 John Pazarzis
 *  Licensed under the GPL License.
 *
 *  This file contains unit and demonstration tests for the ScoreMatrix class,
 *  which implements the Smith-Waterman local alignment algorithm.
 *
 *  Each test checks different aspects such as:
 *    - Matrix initialization and scoring
 *    - Correctness of alignment matrix values
 *    - Retrieval of local alignments
 *    - Handling of various input sequences and scoring parameters
 *
 *  To use: build and run the test binary. Uncomment or add tests in main()
 *  as needed to exercise more features or validate additional cases.
 *
 ******************************************************************************
 */

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

void test2(){
    const std::string s1 = "TCTCGAT";
    const std::string s2 = "GTCTAC";

    ScoreMatrix scoreMatrix(s1, s2, 2, -1, -1);
    const std::string expected_str = scoreMatrix.to_str();
    std::cout << expected_str << std::endl;
}

void test3(){
    const std::string s1 = "AG";
    const std::string s2 = "ATTTTG";

    ScoreMatrix scoreMatrix(s1, s2, 2, -1, -1);
    const std::string expected_str = scoreMatrix.to_str();
    std::cout << expected_str << std::endl;
}

void test4(){
    const std::string s1 = "GCATGC";
    const std::string s2 = "GATTAC";

    ScoreMatrix scoreMatrix(s1, s2, 2, -1, -1);
    const auto alignments = scoreMatrix.getLocalAlignments();
    std::cout << alignments << std::endl;
}

void test5(){
    const std::string s1 = "AGA";
    const std::string s2 = "ATA";

    ScoreMatrix scoreMatrix(s1, s2, 2, -1, -1);
    std::cout << scoreMatrix.to_str() << std::endl;
}


int main(){
    //testScoreMatrix();
    //test2();
    test4();
    //test5();
}



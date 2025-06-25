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

    auto max_position = scoreMatrix.getMaxPosition();
    assert(max_position.first == 5);
    assert(max_position.second == 5);
    std::cout << "Local Alignment:" << scoreMatrix.getLocalAlignment() << std::endl;
    assert(scoreMatrix.getLocalAlignment() == "AC-TC");
}

void test2(){
    const std::string s1 = "TCTCGAT";
    const std::string s2 = "GTCTAC";

    ScoreMatrix scoreMatrix(s1, s2, 2, -1, -1);
    const std::string expected_str = scoreMatrix.to_str();
    std::cout << expected_str << std::endl;
    std::cout << "Local Alignment:" << scoreMatrix.getLocalAlignment() << std::endl;
}

void test3(){
    const std::string s1 = "AG";
    const std::string s2 = "ATTTTG";

    ScoreMatrix scoreMatrix(s1, s2, 2, -1, -1);
    const std::string expected_str = scoreMatrix.to_str();
    std::cout << expected_str << std::endl;
    std::cout << "Local Alignment:" << scoreMatrix.getLocalAlignment() << std::endl;
}

void test4(){
    const std::string s1 = "GCATGC";
    const std::string s2 = "GATTAC";

    ScoreMatrix scoreMatrix(s1, s2, 2, -1, -1);
    const std::string expected_str = scoreMatrix.to_str();
    std::cout << expected_str << std::endl;
    std::cout << "Local Alignment:" << scoreMatrix.getLocalAlignment() << std::endl;
}

void test5(){
    const std::string s1 = "AGA";
    const std::string s2 = "ATA";

    ScoreMatrix scoreMatrix(s1, s2, 2, -1, -1);
    std::cout << scoreMatrix.to_str() << std::endl;

    const std::string retrieved = scoreMatrix.getLocalAlignment();
    const std::string expected = "\nAGA\n*|*\nATA\n";


    assert (retrieved == expected);

    std::cout << "Local Alignment:" << retrieved<< std::endl;
}


int main(){
    //testScoreMatrix();
    //test2();
    test4();
    //test5();
}



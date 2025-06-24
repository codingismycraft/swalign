#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <assert.h>
#include "score_matrix.h"


int main(int argc, char* argv[]) {
    using namespace std;
    assert (argc == 3);
    vector<string> sequences;

    for (int i = 1; i <= 2; ++i) {
        std::ifstream file(argv[i]);
        assert(file.is_open());
        string fulltext;
        std::string line;
        while (std::getline(file, line)) {
            fulltext += line;
        }
        file.close();
        sequences.push_back(fulltext);
    }

    ScoreMatrix score_matrix(sequences[0], sequences[1], 2, 1, -1);
    cout << score_matrix.getLocalAlignment() << endl;

    return 0;
}

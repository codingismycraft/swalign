#ifndef LOCAL_ALIGNMENT_INCLUDED
#define LOCAL_ALIGNMENT_INCLUDED

#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class LocalAlignmentFinder {

    public:
        LocalAlignmentFinder(
                const std::string& s1,
                const std::string& s2,
                int match_score,
                int mismatch_penalty,
                int gap_penalty,
                size_t max_alignments = 10);
        ~LocalAlignmentFinder();

        // Disable copy and move operations
        LocalAlignmentFinder() = delete;
        LocalAlignmentFinder(const LocalAlignmentFinder&) = delete;
        LocalAlignmentFinder& operator=(const LocalAlignmentFinder&) = delete;
        LocalAlignmentFinder(LocalAlignmentFinder&&) = delete;
        LocalAlignmentFinder& operator=(LocalAlignmentFinder&&) = delete;

        // Accessors.
        const std::string& getSequence1() const;
        const std::string& getSequence2() const;
        int getRowCount() const;
        int getColCount() const;
        int getScore(int row, int col) const; // throws std::out_of_range.
        int getMaxScore() const;
        std::string toString() const;
        const std::vector<std::string>& getLocalAlignments() const;
        std::string getLocalAlignmentsAsJson() const;
        size_t getNumberOfAlignments() const;
        void print_matrix() const;
        int getMatrixValue(int row, int col) const;

    private:
        // Private functions.
        void initializeMatrix();
        void traceback(int row, int col, std::string x1, std::string x2, std::string a);
        void processDiagonal(int col, int starting_row);
        int count_anti_diagonal_cells(int anti_diagonal_index);
        void findMaxScores();
        int32_t evaluateScore(const std::string& alighmentStr) const;

    private:
        // Passed from the user.
        const std::string m_sequence1;
        const std::string m_sequence2;
        const int m_rows;
        const int m_cols;
        const int m_match_score;
        const int m_mismatch_penalty;
        const int m_gap_penalty;
        const long long m_matrix_size;
        const size_t m_max_alignments;

        // Keep internal state.
        std::vector<std::pair<int, int>> m_max_positions;
        std::vector<std::string> m_local_alignments;
        int m_max_score;

        // m_matrix is a flat matrix allocated as a contiguous memory block
        // for compatibility with CUDA device memory transfers.
        // Do NOT use std::vector or smart pointers here.
        int* m_matrix;
};

std::ostream& operator<<(std::ostream& os, const LocalAlignmentFinder& obj);


#endif // LOCAL_ALIGNMENT_INCLUDED

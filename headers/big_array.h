#ifndef BIG_ARRAY_INCLUDED
#define BIG_ARRAY_INCLUDED

#include <string>
#include <cstdint>
#include <memory>



class BigArray{
    private:

        const std::string m_filename;

        uint64_t m_rows;
        uint64_t m_cols;
        uint64_t m_file_size;   // Use uint64_t for huge file sizes!
        int m_fd;

        void* m_mmapped_ptr;
        int32_t* m_data;

        // Private constructor to enforce use of factory methods
        BigArray(const std::string& filename, uint64_t rows, uint64_t cols);
        BigArray(const std::string& filename);


        // Disable copy and move semantics
        BigArray(const BigArray&) = delete;
        BigArray& operator=(const BigArray&) = delete;
        BigArray(BigArray&&) = delete;
        BigArray& operator=(BigArray&&) = delete;

    public:

        static std::unique_ptr<BigArray> make_new(const std::string& filename, uint64_t rows, uint64_t  cols);
        static std::unique_ptr<BigArray> load(const std::string& filename);

        ~BigArray();

        int32_t get(uint64_t row, uint64_t col) const;
        void set(uint64_t row, uint64_t col, int value);
        const std::string& get_filename() const;
};




#endif // BIG_ARRAY_INCLUDED

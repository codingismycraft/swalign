#include <iostream>
#include <utils.h>

#include <filesystem>
#include <string>
#include <fstream>

std::filesystem::path get_source_dir() {
    std::filesystem::path file_path(__FILE__);
    return file_path.parent_path();
}

int main(){
    std::cout << "Testing the utils component" << std::endl;
    const std::filesystem::path source_dir = get_source_dir();
    std::cout << source_dir << std::endl;

     // Create the path to your file
    std::filesystem::path file_path = std::filesystem::current_path() / "tests" / "data" / "seq1.txt";

    // Open the file
    std::ifstream infile(file_path);

    if (!infile) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return 1;
    }

    // Read and print the file line by line
    std::string line;
    while (std::getline(infile, line)) {
        std::cout << line << '\n';
    }
}

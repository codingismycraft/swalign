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

     // Create the path to your file
    std::filesystem::path file_path = std::filesystem::current_path() / "tests" / "data" / "seq1.txt";
    std::string chromosome = readFastaSequence(file_path.string());
    std::cout << chromosome << std::endl;

}

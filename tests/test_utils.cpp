#include <iostream>
#include <utils.h>

#include <filesystem>
#include <string>
#include <fstream>

std::filesystem::path get_source_dir() {
    std::filesystem::path file_path(__FILE__);
    return file_path.parent_path();
}

void test_generating_random_name() {
    std::string name = generate_random_name();
    std::cout << "Generated random name: " << name << std::endl;
    std::cout <<"Successfully generated a random name of length 10." << std::endl;
}

int main(){
    std::cout << "Testing the utils component" << std::endl;
    const std::filesystem::path source_dir = get_source_dir();

     // Create the path to your file
    std::filesystem::path file_path = std::filesystem::current_path() / "tests" / "data" / "seq1.txt";
    std::string chromosome = readFastaSequence(file_path.string());
    std::cout << chromosome << std::endl;
    test_generating_random_name();

}

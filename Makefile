# Compiler and flags
#
CXX = g++
CXXFLAGS = -std=c++20 -Wall  -g -Iheaders/

SRC_DIR = ./src
TEST_DIR = ./tests
BIN_DIR = ./bin
# Source and test files
SRC = $(SRC_DIR)/score_matrix.cpp
TEST = $(TEST_DIR)/test_score_matrix.cpp

# Output binary
TARGET = test

$(TARGET): $(SRC) $(TEST)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(SRC) $(TEST) -o $(BIN_DIR)/$(TARGET)

clean:
	rm -f $(BIN_DIR)/$(TARGET)


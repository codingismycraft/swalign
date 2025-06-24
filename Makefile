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
HEADERS = headers/score_matrix.h

$(BIN_DIR)/test: $(HEADERS)  $(SRC) $(TEST)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(SRC) $(TEST) -o $(BIN_DIR)/test

test: $(BIN_DIR)/test
	$(BIN_DIR)/test

main: $(HEADERS)  $(SRC) $(SRC_DIR)/main.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(SRC) $(SRC_DIR)/main.cpp -o $(BIN_DIR)/main


clean:
	rm -f $(BIN_DIR)/*


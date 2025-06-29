###############################################################################
#
# Copyright (c) 2025 John Pazarzis
#
# Licensed under the GPL License.
#
# Makefile for building swalign and its tests in both debug and release modes.
#
# Usage Examples:
#   Build swalign in debug mode (default)
#   make swalign
#
#   Build swalign in release mode
#   make swalign BUILD=release
#
#   Build and run tests in debug mode (default)
#   make test
#
#   Build and run tests in release mode
#   make test BUILD=release
#
#   Clean all binaries
#   make clean
#
###############################################################################

# Compiler and flags
CXX = g++
SRC_DIR = ./src
TEST_DIR = ./tests
HEADERS =  -Iheaders -I${RAPIDJSON_HOME}

# Default build is debug
BUILD ?= debug


.PHONY: all clean debug release test_score_matrix test_utils swalign


ifeq ($(BUILD),release)
    CXXFLAGS = -std=c++20 -Wall -O3 -DNDEBUG ${HEADERS}
    BIN_DIR = ./bin/release
    OBJDIR = ./obj/release
else
    CXXFLAGS = -std=c++20 -Wall -g ${HEADERS}
    BIN_DIR = ./bin/debug
    OBJDIR = ./obj/debug
endif


clean:
	echo $(OBJDIR)
	rm -rf $(OBJDIR)
	rm -rf $(BIN_DIR)


$(OBJDIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@


$(BIN_DIR)/test_score_matrix: ./headers/score_matrix.h $(OBJDIR)/score_matrix.o $(TEST_DIR)/test_score_matrix.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJDIR)/score_matrix.o $(TEST_DIR)/test_score_matrix.cpp


test_score_matrix: $(BIN_DIR)/test_score_matrix
	$(BIN_DIR)/test_score_matrix


$(BIN_DIR)/test_utils: ./headers/utils.h $(OBJDIR)/utils.o $(TEST_DIR)/test_utils.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJDIR)/utils.o $(TEST_DIR)/test_utils.cpp


test_utils: $(BIN_DIR)/test_utils
	$(BIN_DIR)/test_utils


$(BIN_DIR)/swalign: ./headers/score_matrix.h $(OBJDIR)/score_matrix.o $(OBJDIR)/utils.o $(OBJDIR)/swalign.o
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(BIN_DIR)/swalign $(OBJDIR)/score_matrix.o $(OBJDIR)/utils.o $(OBJDIR)/swalign.o


swalign: $(BIN_DIR)/swalign

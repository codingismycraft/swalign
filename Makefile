###############################################################################
# swalign - Smith-Waterman Local Alignment Tool
#
# Makefile for building swalign and its tests in both debug and release modes.
#
# Usage Examples:
#   # Build swalign in debug mode (default)
#   make swalign
#
#   # Build swalign in release mode
#   make swalign BUILD=release
#
#   # Build and run tests in debug mode (default)
#   make test
#
#   # Build and run tests in release mode
#   make test BUILD=release
#
#   # Clean all binaries
#   make clean
#
###############################################################################

# Compiler and flags
CXX = g++
SRC_DIR = ./src
TEST_DIR = ./tests
BIN_DIR = ./bin
HEADERS = headers/score_matrix.h
SRC = $(SRC_DIR)/score_matrix.cpp
TEST = $(TEST_DIR)/test_score_matrix.cpp

# Default build is debug
BUILD ?= debug

# Flags for debug and release
CXXFLAGS_DEBUG = -std=c++20 -Wall -g -Iheaders/
CXXFLAGS_RELEASE = -std=c++20 -Wall -O3 -DNDEBUG -Iheaders/

ifeq ($(BUILD),release)
    CXXFLAGS = $(CXXFLAGS_RELEASE)
    BIN_SUFFIX =
else
    CXXFLAGS = $(CXXFLAGS_DEBUG)
    BIN_SUFFIX = _debug
endif

.PHONY: all debug release test clean

all: swalign$(BIN_SUFFIX) test$(BIN_SUFFIX)

debug:
	$(MAKE) all BUILD=debug

release:
	$(MAKE) all BUILD=release

$(BIN_DIR)/test$(BIN_SUFFIX): $(HEADERS) $(SRC) $(TEST)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(SRC) $(TEST) -o $(BIN_DIR)/test$(BIN_SUFFIX)

test$(BIN_SUFFIX): $(BIN_DIR)/test$(BIN_SUFFIX)
	$(BIN_DIR)/test$(BIN_SUFFIX)

test: test$(BIN_SUFFIX)

$(BIN_DIR)/swalign$(BIN_SUFFIX): $(HEADERS) $(SRC) $(SRC_DIR)/swalign.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(SRC) $(SRC_DIR)/swalign.cpp -o $(BIN_DIR)/swalign$(BIN_SUFFIX)

swalign$(BIN_SUFFIX): $(BIN_DIR)/swalign$(BIN_SUFFIX)

swalign: swalign$(BIN_SUFFIX)

clean:
	rm -f $(BIN_DIR)/*

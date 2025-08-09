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
#   make test BUILD=releas./e
#
#   Clean all binaries
#   make clean
#
###############################################################################

# Compiler and flags
CXX = g++
SRC_DIR = ./src
TEST_DIR = ./tests
HEADERS =  -I./headers -I${RAPIDJSON_HOME}

CUDA_CXX = nvcc

# Default build is debug
BUILD ?= debug


.PHONY: all clean debug release test_score_matrix test_utils \
	swalign help cuda_sample smith_waterman1 tags \
	data_input test_big_array


help:  ## Show this help and exit
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

ifeq ($(BUILD),release)
    CXXFLAGS = -std=c++20 -Wall -O3 -DNDEBUG ${HEADERS}
    BIN_DIR = ./bin/release
    OBJDIR = ./obj/release
	NVCCFLAGS = -Xcompiler -Wall ${HEADERS}
else
    CXXFLAGS = -std=c++20 -Wall -g ${HEADERS}
    BIN_DIR = ./bin/debug
    OBJDIR = ./obj/debug
	NVCCFLAGS = -g -G -O0 -Xcompiler -Wall ${HEADERS}
endif


clean:
	rm -rf ./obj
	rm -rf ./bin


$(OBJDIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@


$(OBJDIR)/big_array.o: $(SRC_DIR)/big_array.cpp ./headers/big_array.h
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build rule for CUDA source files
$(OBJDIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJDIR)
	$(CUDA_CXX) $(NVCCFLAGS) -c $< -o $@


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


$(BIN_DIR)/test_big_array: ./headers/big_array.h $(OBJDIR)/utils.o $(OBJDIR)/big_array.o $(TEST_DIR)/test_big_array.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJDIR)/utils.o $(OBJDIR)/big_array.o $(TEST_DIR)/test_big_array.cpp


test_big_array: $(BIN_DIR)/test_big_array
	$(BIN_DIR)/test_big_array

$(BIN_DIR)/swalign: ./headers/score_matrix.h $(OBJDIR)/score_matrix.o $(OBJDIR)/utils.o $(OBJDIR)/swalign.o
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJDIR)/score_matrix.o $(OBJDIR)/utils.o $(OBJDIR)/swalign.o


swalign: $(BIN_DIR)/swalign

$(BIN_DIR)/cuda_sample: $(SRC_DIR)/cuda_sample.cu
	@mkdir -p $(BIN_DIR)
	$(CUDA_CXX) -g -G $(SRC_DIR)/cuda_sample.cu -o $(BIN_DIR)/cuda_sample


cuda_sample: $(BIN_DIR)/cuda_sample
	$(BIN_DIR)/cuda_sample


$(BIN_DIR)/smith_waterman1: $(SRC_DIR)/smith_waterman1.cu
	@mkdir -p $(BIN_DIR)
	$(CUDA_CXX) -g -G -O0 $(SRC_DIR)/smith_waterman1.cu -o $(BIN_DIR)/smith_waterman1


smith_waterman1: $(BIN_DIR)/smith_waterman1
	$(BIN_DIR)/smith_waterman1


$(BIN_DIR)/data_input: $(SRC_DIR)/data_input.cu $(OBJDIR)/local_alignment.o $(OBJDIR)/aligner.o $(OBJDIR)/big_array.o $(OBJDIR)/utils.o
	@mkdir -p $(BIN_DIR)
	$(CUDA_CXX) $(NVCCFLAGS) $(SRC_DIR)/data_input.cu -o $(BIN_DIR)/data_input $(OBJDIR)/local_alignment.o $(OBJDIR)/aligner.o $(OBJDIR)/big_array.o $(OBJDIR)/utils.o


data_input: $(BIN_DIR)/data_input
	$(BIN_DIR)/data_input

tags:
	ctags -R -f tags .  ${RAPIDJSON_HOME}

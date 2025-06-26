# swalign

**Smith-Waterman Local Alignment Tool**

![License: GPL](https://img.shields.io/badge/License-GPL-blue.svg)

## Overview

**swalign** is a robust C++ command-line tool for computing optimal local alignments between two biological (or arbitrary) sequences using the [Smith-Waterman algorithm](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm). It is designed for researchers, students, and developers who need fast, reliable, and customizable local sequence alignment.


## Installation

To use **swalign**, you need to have a C++ compiler installed on your system.
The project uses a `Makefile` for building the application, which simplifies
the compilation process.

```
cd <your-repos-directory>
git clone https://github.com/codingismycraft/swalign.git
git clone https://github.com/Tencent/rapidjson.git
```

Add the following line to your `.bashrc` or `.zshrc`

```
export RAPIDJSON_HOME=<your-repos-directory>/rapidjson/include
```

## Test

From the swalign directory, run the following command to execute the unit tests:

```sh

make test
```

## Building swalign

The project uses a `Makefile` to support both **debug** and **release** builds. Binaries are placed in `bin/debug/` and `bin/release/` respectively.

### Build swalign (debug mode, default)

```sh
make swalign
# Binary: bin/debug/swalign_debug
```

### Build swalign (release mode)

```sh
make swalign BUILD=release
# Binary: bin/release/swalign
```

### Build and run tests (debug mode)

```sh
make test
# Runs: bin/debug/test_debug
```

### Build and run tests (release mode)

```sh
make test BUILD=release
# Runs: bin/release/test
```

### Clean all build artifacts

```sh
make clean
```

## Features

- **Local alignment** using the Smith-Waterman algorithm
- Customizable match score, mismatch penalty, and gap penalty
- Detailed output of the scoring matrix and all optimal local alignments
- Easy-to-read command-line interface
- Well-tested with unit and demonstration tests

## Usage

```sh
swalign <seq1_file> <seq2_file> [options]
```

### Arguments

- `<seq1_file>`: File containing the first sequence (required)
- `<seq2_file>`: File containing the second sequence (required)

### Options

- `-m <int>`: Match score (default: 2)
- `-x <int>`: Mismatch penalty (default: -1)
- `-g <int>`: Gap penalty (default: -1)
- `-h, --help`: Print help message and exit

### Notes

- All score and penalty values must be integers between -10 and 10.
- Input files must exist and be readable.

### Example

```sh
./swalign seq1.txt seq2.txt -m 2 -x -1 -g -1
```


## Project Structure

```
.
├── bin/           # Compiled binaries (debug and release)
├── headers/       # C++ header files
│   └── score_matrix.h
├── src/           # Source files
│   ├── score_matrix.cpp
│   └── swalign.cpp
├── tests/         # Unit and demonstration test files
│   └── test_score_matrix.cpp
├── Makefile
└── README.md
```

## License

This project is licensed under the [GPL License](LICENSE).

---


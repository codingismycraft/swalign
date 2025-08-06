/**
 *
 * Copyright (c) 2025 John Pazarzis
 *
 * Licensed under the GPL License.
 */

#ifndef __SWALIGN_UTILS_INCLUDED__
#define __SWALIGN_UTILS_INCLUDED__

#include <string>

/**
 * @brief Reads a DNA or protein sequence from a FASTA file, concatenating all
 * non-header lines into a single string.
 *
 * @param fasta_path The path to the FASTA file to read.  @return A std::string
 * containing the concatenated, trimmed sequence from the file.
 *
 * @throws std::runtime_error If the file cannot be opened.
 *
 * @note Only the first header line (starting with '>') is skipped; all later
 * lines are assumed to be part of the sequence.  @note The function is robust
 * to extra whitespace within sequence lines, but assumes a standard FASTA
 * format.
 *
 * @example
 * // Given a FASTA file "example.fasta" with:
 * // >seq1
 * // ACTG
 * // TGCA
 * // The function returns "ACTGTGCA".
 */
std::string readFastaSequence(const std::string& fasta_path);


std::string generate_random_name();

#endif // __SWALIGN_UTILS_INCLUDED__

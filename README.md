# Local Align

A C++ implementation of the Smith-Waterman algorithm for local sequence
alignment.

# Traceback algorithm

- Start from the cell with the highest score and trace back to the cell with a
score of 0.

- Keep track of the current row, col.

- Create two new strings (x1, x2) each of them will contain the aligned
  sequences for the two input sequences.

  x1 => the horizontal
  x2 => the vertical

- Create a new string called alignment (a) that will contain the aligned
  sequences as a set of the following characters:
  j
  * (match)
  | mismatch
  ' ' (gap).

- Move the current row and column according to the following rules:

  - If the current cell is the result of a match, move to the upper left cell
    (diagonal move).
  - If the current cell is the result of a mismatch, move to the upper left cell
    (diagonal move).
  - If the current cell is the result of a gap in the up sequence, move to the
    left cell.
  - If the current cell is the result of a gap in the left sequence, move to
    the upper cell.

- if the score of the current cell is 0, stop the traceback.

There are the following possible moves:

### Move 1: Matching char
-  The current cell is the result of a match (diagonal move)

in this case append the current (matching) character to the "begining" of each
of x1 and x2, and move to the cell in the upper left corner.

append the * to the alignment string (a)

### Move 2: Mismatching char
- The current cell is the result of a mismatch (diagonal move). Add the
  corresponding characters to the "begining" of each of x1 and x2, and move to
  the cell in the upper left corner.

append the | to the alignment string (a)

### Move 3: Gap in the up sequence

Add a space to the begining of the alignment string (a)
Add an underscore to the begining of x2
Add the character from the horizontal sequence to the begining of x1.

### Move 4: Gap in the left sequence

Add a space to the begining of the alignment string (a)
Add an underscore to the begining of x1
Add the character from the horizontal sequence to the begining of x2.

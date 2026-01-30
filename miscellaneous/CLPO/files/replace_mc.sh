#!/bin/bash

# A script to replace all occurrences of 'Beta' with 'Alpha' in a file.
#
# Usage: ./replace_text.sh <filename>
#
# Arguments:
#   <filename>: The path to the file you want to modify.

# Check if a filename was provided as an argument.
if [ -z "$1" ]; then
  echo "Error: Please provide a filename as an argument."
  echo "Usage: ./replace_text.sh <filename>"
  exit 1
fi

FILE="$1"

# Check if the file exists.
if [ ! -f "$FILE" ]; then
  echo "Error: File '$FILE' not found."
  exit 1
fi

#echo "Replacing 'BETA' with 'ALPHA' in file: $FILE"

# Use sed to perform a global in-place replacement.
# 's/BETA/ALPHA/g' is the substitution command:
#   's' stands for substitute.
#   'g' stands for global, meaning all occurrences on each line will be replaced.
# -i flag modifies the file in place.
sed -i 's/Beta/Alpha/g' "$FILE"

#echo "Replacement complete."

#!/bin/bash

# A script to convert a codebase into a single text file with hierarchy,
# ignoring specified files and directories.

# Set the output file name
OUTPUT_FILE="codebase.txt"

# Set the root directory of your codebase
# By default, it uses the current directory where the script is run
CODE_ROOT_DIR="."

# Define the files and directories to ignore 🚫
# IMPORTANT: Directories MUST end with a trailing slash to be correctly ignored.
# This tells the script's logic to use the efficient -prune option.
EXCLUDE_LIST=(
    ".git/"
    "node_modules/"
    "__pycache__/"
    "dist/"
    "*.log"
    "*.pyc"
    "*.avi"
    "build/"
    "install/"
    "log"
    "README.md"
)

# Clear the output file if it exists
> "$OUTPUT_FILE"

echo "Generating codebase hierarchy..."
# Convert the bash array to a format 'tree' understands: 'pattern1|pattern2|...'
EXCLUDE_STRING=$(printf "%s|" "${EXCLUDE_LIST[@]}")
EXCLUDE_STRING=${EXCLUDE_STRING%?} # Remove the last pipe character

# Use 'tree' with the -I option to exclude directories and files
tree -a -I "$EXCLUDE_STRING" "$CODE_ROOT_DIR" >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "==========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "Concatenating file contents..."

# Build the find command dynamically to apply exclusions correctly
FIND_CMD="find \"$CODE_ROOT_DIR\""

# Add the exclusion logic for each item
for item in "${EXCLUDE_LIST[@]}"; do
    if [[ "$item" == */ ]]; then
        # This is a directory, use -prune to skip it and its contents
        FIND_CMD+=" -path \"$CODE_ROOT_DIR/${item%?}\" -prune -o"
    else
        # This is a file, use ! -name to exclude it
        FIND_CMD+=" -name \"$item\" -prune -o"
    fi
done

# Add the final -type f to select only files that haven't been pruned
FIND_CMD+=" -type f -print0"

# Use eval to execute the dynamically built find command, and pipe to while loop
eval "$FIND_CMD" | while IFS= read -r -d $'\0' file; do
    echo "--- File: $file ---" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE" # Add a newline for separation
done

echo "Done! Codebase has been exported to $OUTPUT_FILE"

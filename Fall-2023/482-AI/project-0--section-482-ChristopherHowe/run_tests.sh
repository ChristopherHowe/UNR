# List all test files in the current directory using ls and grep
TEST_FILES=$(ls -a | grep -E '\.tictac_test_[0-9]+\.py')
echo test files: $TEST_FILES
# Loop through the test files and run them
for TEST_FILE in $TEST_FILES; do
    # Extract the filename from the ls output
    FILENAME=$(echo "$TEST_FILE" | awk '{print $9}')
    
    # Ensure the file is not a directory
    if [ -f "$TEST_FILE" ]; then
        echo "Running test: $TEST_FILE"
        python3 "$TEST_FILE"
        echo "==============================================="
    fi
done





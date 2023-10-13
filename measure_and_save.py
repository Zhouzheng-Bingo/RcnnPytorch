
import sys
import subprocess

def run_and_save_output(file_path, output_file, times=50):
    python_path = sys.executable
    with open(output_file, 'a') as f:
        for _ in range(times):
            # Run the script and capture the output using the current Python interpreter
            result = subprocess.run([python_path, file_path], capture_output=True, text=True)
            # Write the captured output to the specified file
            f.write(result.stdout)
            f.write("\n" + "-"*50 + "\n")  # Add a separator between runs

# Script execution starts here
if __name__ == "__main__":
    script_path = "per_layer_time_env.py"  # Assuming it's in the same directory
    output_filename = "results.txt"
    run_and_save_output(script_path, output_filename)

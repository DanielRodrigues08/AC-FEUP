import os
import subprocess
import argparse


def main(path):
    # Loop through each file in the directory
    for file in os.listdir(path):
        # Check if the file is a Jupyter notebook
        if file.endswith('.ipynb'):
            # Convert the notebook to a Python script using nbconvert
            subprocess.run(['jupyter', 'nbconvert', '--to', 'script', os.path.join(path, file)])
            # Run the Python script using subprocess
            subprocess.run(['python', os.path.join(path, file.replace('.ipynb', '.py'))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='cleaning')
    args = parser.parse_args()
    main(args.path)

import shutil
import os

def copy_files():
    # List of files to copy
    files_to_copy = ['00023598.npz', '00072256.npz', '00054350.npz']

    # Source and destination directories
    src_dir = '/eos/experiment/shadows/user/flstumme/ai/data/bend_h/prepared'  # Replace with your source folder path
    dst_dir = '/eos/experiment/shadows/user/flstumme/ai/data/bend_h/test'  # Replace with your destination folder path

    # Ensure destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    # Copy files
    for file in files_to_copy:
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(dst_dir, file)
        
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            print(f'Copied {file} to {dst_dir}')
        else:
            print(f'{file} does not exist in the source directory')

if __name__ == '__main__':
    copy_files()
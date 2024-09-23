import shutil
import os

def copy_files():
    # List of files to copy
    files_to_copy = ['00023598.npz', '00072256.npz', '00054350.npz', '00037585.npz', '00009908.npz', '00068424.npz', '00027975.npz', '00040393.npz', '00039197.npz', '00015875.npz', '00029436.npz', '00026436.npz', '00033214.npz', '00009755.npz', '00008658.npz', '00006942.npz', '00054143.npz', '00035430.npz', '00037533.npz', '00044505.npz', '00057400.npz', '00061952.npz', '00056837.npz', '00029886.npz', '00022309.npz', '00010673.npz', '00066642.npz', '00010671.npz', '00023626.npz', '00019811.npz', '00012883.npz', '00016386.npz', '00054890.npz', '00043631.npz', '00043253.npz', '00034534.npz', '00057733.npz', '00008426.npz', '00023159.npz', '00056649.npz', '00019770.npz', '00073375.npz', '00065686.npz', '00047942.npz', '00034468.npz', '00020821.npz', '00023835.npz', '00020214.npz', '00018924.npz', '00014001.npz']

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
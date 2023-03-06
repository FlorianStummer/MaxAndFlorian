
import numpy as np
import os
import preparation_functions as pf
import time



def get_args() -> dict:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='the name of the dataset to prepare in the input directory. Note that this has to be unzipped.')
    parser.add_argument('-i', '--in_dir', type=str, help='Directory from which the raw csvs are loaded.', default="../../data/raw")
    parser.add_argument('-o', '--out_dir', type=str, help='Directory to which the prepared dataset is written.', default="../../data/prepared")
    parser.add_argument('-m','--metadata_file', type=str, help='the name of the metadatafile. default is "metadata.csv"', default="metadata.csv")
    parser.add_argument('-f','--preparation_function',type=str, help='the function from src/data_preparation/preparation_functions.py that is used to prepare the data"', default="create_unet_images" )
    args = parser.parse_args()

    if "." in args.name or "/" in args.name:
        raise ValueError("invalid filename. May not contain '.' or '/' ")
    
    return args



if __name__ == "__main__":
    args = get_args()
    #
    input_path = os.path.abspath(os.path.join(args.in_dir,args.name))
    output_path = os.path.abspath(os.path.join(args.out_dir,args.name))
    metadata_path = os.path.abspath(os.path.join(input_path,args.metadata_file))

    print("reading input data from", input_path)
    print("writing new dataset to", output_path)
    print("expecting metadata in", metadata_path)
    print("using the function", args.preparation_function," from src/data_preparation/preparation_functions.py to prepare the data.")

    pref_fun = getattr(pf,args.preparation_function)

    filelist = sorted([f for f in os.listdir(input_path) if not f == args.metadata_file])

    metainfo_file = np.genfromtxt(metadata_path, delimiter=',', unpack=True)
    metainfo = np.swapaxes(metainfo_file[0:,1:], 0, 1)
    metainfo_header = [l for l in open(metadata_path,"r")][0].split(",")
    print(metainfo_header)

    if not metainfo.shape[0] == len(filelist):
        raise ValueError("number of metainfos does not match up with number of files. \
                         There are " + str(metainfo.shape[0]) + " entries and " + str(len(filelist)) + " files. \
                         Consider hidden files.")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    time_accum = []
    total_samples = len(filelist)

    for i, cur_file in enumerate(filelist):
        t_0 = time.time()
        Bx,By = np.genfromtxt(os.path.join(input_path,cur_file), delimiter=',', unpack=True)[2:4]
        Bx = np.resize(Bx[1:], (81,121)).T
        By = np.resize(By[1:], (81,121)).T
        Bx[np.abs(Bx) < 0.01] = 0
        By[np.abs(By) < 0.01] = 0
        tar = np.flip(np.stack([Bx, By], axis=0), axis=1)

        inp = pref_fun(metainfo[i][1:])

        inp  = inp[:,:120,:80]
        tar = tar[:,:120,:80]

        np.savez_compressed(os.path.join(output_path,str(i)+".npz"),input=inp,target=tar)

        time_accum.append(time.time()-t_0)

        if i % 10 == 0:
            running_avg = np.array(time_accum).mean()
            print("completed {0:06d}/{1} samples. ETA: {2:.0f} seconds. Current running avg: {3:.3f} seconds".format(i,total_samples,(total_samples-i)*running_avg,running_avg))

    np.savez_compressed(os.path.join(output_path,"meta.npz"),metainfos=metainfo)







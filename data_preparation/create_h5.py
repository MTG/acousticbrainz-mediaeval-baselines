import h5py
import numpy as np

def prepare_set(input_path, dataset_name, set_name, compute_y=True):
    """Create the dataset files for Tartarus given the CSVs of features from the
    acousticbrainz dataset

    Parameters
    ----------
    input_path : str
        The path to the feature csv files
    dataset_name : str
        The name of the output dataset
    set_name : str
        The name of the set (train, test, val)
    compute_y : bool, optional
        If true write a numpy matrix with the ground truth
    """
    if compute_y:
        f=open(input_path+".features.clean.std.csv")
    else:
        f=open(input_path+".clean.std.csv")
    lines = f.readlines()[1:]
    n_items = len(lines)
    n_features = len(lines[0].strip().split(",")) - 1
    f = h5py.File('data-genres/patches/patches_%s_%s_%sx%s.hdf5' % (set_name,dataset_name,1,1),'w')
    x_dset = f.create_dataset("features", (n_items,n_features), dtype='f')
    i_dset = f.create_dataset("index", (n_items,), maxshape=(n_items,), dtype='S36')
    #y_dset = f.create_dataset("targets", (n_items,genres), dtype='i')
    if compute_y:
        fg=open(input_path+".genres.csv")
        genres = fg.readlines()[1:]
        n_genres = len(genres[0].strip().split(",")) - 1
        y_dset = np.zeros((n_items,n_genres))
        print(y_dset.shape)
        print(n_genres)
        k=0
        id2gt = dict()
        for genre in genres:
            items_genres = genre.strip().split(",")
            id2gt[items_genres[0]] = [int(k) for k in items_genres[1:]]
    
    itemset = []
    for t,line in enumerate(lines):
        items = line.strip().split(",")
        x_dset[t,:] = [float(k) for k in items[1:]]
        i_dset[t] = items[0]
        if compute_y:
            y_dset[t,:] = np.array(id2gt[items[0]])
        itemset.append(items[0])
        if t%1000==0:
            print(t)

    print(x_dset.shape)
    if compute_y:
        np.save('data-genres/splits/y_%s_class_%s_%s.npy' % (set_name,n_genres,dataset_name),y_dset)
    fw = open('data-genres/splits/items_index_%s_%s.tsv' % (set_name,dataset_name),'w')
    fw.write("\n".join(itemset))

if __name__ == "__main__":
    prepare_set("../processed/train/allmusic-train-test","allmusic","val")
    prepare_set("../processed/train/allmusic-train-train","allmusic","train")
    prepare_set("../processed/validation/allmusic-validation","allmusic","test")
    prepare_set("../processed/train/discogs-train-test","discogs","val")
    prepare_set("../processed/train/discogs-train-train","discogs","train")
    prepare_set("../processed/validation/discogs-validation","discogs","test")
    prepare_set("../processed/train/lastfm-train-test","lastfm","val")
    prepare_set("../processed/train/lastfm-train-train","lastfm","train")
    prepare_set("../processed/validation/lastfm-validation","lastfm","test")
    prepare_set("../processed/train/tagtraum-train-test","tagtraum","val")
    prepare_set("../processed/train/tagtraum-train-train","tagtraum","train")
    prepare_set("../processed/validation/tagtraum-validation","tagtraum","test")
    prepare_set("../processed/test/test_features-allmusic","allmusic","realtest",compute_y=False)
    prepare_set("../processed/test/test_features-discogs","discogs","realtest",compute_y=False)
    prepare_set("../processed/test/test_features-lastfm","lastfm","realtest",compute_y=False)
    prepare_set("../processed/test/test_features-tagtraum","tagtraum","realtest",compute_y=False)

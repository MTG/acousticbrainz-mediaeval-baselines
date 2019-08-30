import csv
from argparse import ArgumentParser


def load_genres(gt_file):
    """
    Loads a list of all genres present in a ground-truth CSV file.

    :param gt_file: ground-truth file with genre annotations
    :return: sorted list of genres
    """
    genres_all = set()
    with open(gt_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            mbid = row[0]
            genres = [g for g in row[2:] if g != '']
            genres_all.update(genres)
    return sorted(list(genres_all))


def convert(gt_file, out_file, genres=None):
    """
    Convert ground truth into matrix format and store to file
    
    :param gt_file: ground-truth file with genre annotations
    :param out_file: output CSV file to store the resulting matrix
    :param genres: list of genre strings in the ground truth
    """
    if not genres:
        genres = load_genres(gt_file)

    with open(gt_file, 'r') as f_in:
        reader = csv.reader(f_in, delimiter='\t')
        header = next(reader)
        
        with open(out_file, 'w') as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(['recordingmbid'] + genres)

            for row in reader:
                mbid = row[0]
                mbid_genres = [g for g in row[2:] if g != '']
                result = [0] * len(genres)
                for g in mbid_genres:
                    result[genres.index(g)] = 1
                writer.writerow([mbid] + result)
    return


if __name__ == '__main__':
    parser = ArgumentParser(description = "Convert genre ground truth from the original TSV format to a CSV binary matrix")
    parser.add_argument("input_tsv", help="input TSV file with a genre ground truth")
    parser.add_argument('output_csv', help="output CVS with a binary genre matrix")

    args = parser.parse_args()

    convert(args.input_tsv, args.output_csv)

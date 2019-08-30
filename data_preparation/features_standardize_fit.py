import csv, sys
import pickle
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler

def compute(input_csv, output_model, chunk_size=10000):
    with open(input_csv, 'r') as fi:
        reader = csv.reader(fi)
        header = next(reader)

        if header.index('recordingmbid') != 0:
            print("Wrong header. Expected the 'recordingmbid' field as the first column")
            sys.exit()

        scaler = StandardScaler()

        progress = 0
        chunk_features = []
        for row in reader:
            mbid = row[0]
            row_features = [float(x) for x in row[1:]]
            chunk_features.append(row_features)

            if len(chunk_features) == chunk_size:
                scaler.partial_fit(chunk_features)
                progress += 1
                chunk_features = []
                print("Loaded chunks: %d (%d recordings)" % (progress, progress * chunk_size))

        # Last chunk
        if len(chunk_features):
            scaler.partial_fit(chunk_features)

        print("All done. Saving scaler model to file %s" % output_model)
        with open(output_model, 'wb') as output:
            pickle.dump((header[1:], scaler), output)

    return


if __name__ == '__main__':
    parser = ArgumentParser(description = "Fit a standardization model for a CSV feature matrix")
    parser.add_argument("input_csv", help="input CSV feature matrix")
    parser.add_argument("output_model", help="file to store the fitted model")

    args = parser.parse_args()

    compute(args.input_csv, args.output_model)

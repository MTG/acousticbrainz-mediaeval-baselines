import csv, sys
import pickle
from argparse import ArgumentParser

def compute(input_csv, output_csv, model):
    # Load model
    with open(model, 'rb') as f:
        (header_reference, scaler) = pickle.load(f)

    with open(input_csv, 'r') as fi:
        reader = csv.reader(fi)
        header = next(reader)

        if header.index('recordingmbid') != 0:
            print("Wrong header. Expected the 'recordingmbid' field as the first column")
            sys.exit()

        if header[1:] != header_reference:
            print("Wrong header. To use the model, the expected header features are", header_reference)


        with open(output_csv, 'w') as fo:
            writer = csv.writer(fo)
            writer.writerow(header)

            for row in reader:
                mbid = row[0]
                row_features = [float(x) for x in row[1:]]
                norm_features = list(scaler.transform([row_features])[0])
                writer.writerow([mbid] + norm_features)
    return


if __name__ == '__main__':
    parser = ArgumentParser(description = "Standardize features in a CSV matrix")
    parser.add_argument("input_csv", help="input CSV feature matrix")
    parser.add_argument("output_csv", help="output CSV matrix with standardized features")
    parser.add_argument("model", help="model file (produced by features_standardize_fit.py)")

    args = parser.parse_args()

    compute(args.input_csv, args.output_csv, args.model)

import csv, sys
from argparse import ArgumentParser


def extract(csv_file, mbids_csv, out_file):
    # Load subset recording MBIDs
    mbids_all = set()
    with open(mbids_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        if header[0] != "recordingmbid":
            print("Wrong header. Expected the 'recordingmbid' field")
            sys.exit()
        for row in reader:
            mbid = row[0]
            mbids_all.add(mbid)
    print("Loaded %d recording MBIDs" % len(mbids_all))

    # Process the input csv file
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        with open(out_file, 'w') as fw:
            writer = csv.writer(fw)
            writer.writerow(header)
            for row in reader:
                mbid = row[0]
                if mbid in mbids_all:
                    writer.writerow(row)
    return


if __name__ == '__main__':
    parser = ArgumentParser(description = "Extract a subset of the CSV feature (genre) matrix for given recording MBIDs")
    parser.add_argument("input_csv", help="input CSV file with a feature (genre) matrix")
    parser.add_argument("subset_mbids_csv", help="input CSV file with recording MBIDs to extract in the first column")
    parser.add_argument('output_mbids_csv', help="output CSV with the resulting sub-matrix")

    args = parser.parse_args()

    extract(args.input_csv, args.subset_mbids_csv, args.output_mbids_csv)

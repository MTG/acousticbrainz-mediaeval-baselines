import csv, sys
from argparse import ArgumentParser


def clean_mbid(mbid):
    # Strip down MBID value from the filename
    mbid = mbid.split('/')[-1]
    if mbid.endswith('.json'):
        mbid = mbid.split('.')[0]
        return mbid
    else:
        print("Wrong MBID: %s" % mbid)
        return None


key_values = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
encoding = {
    'tonal.key_key': key_values,    
    'tonal.key_scale': ['major', 'minor'],
    'tonal.chords_key': key_values,
    'tonal.chords_scale': ['major', 'minor']
    }


def encode(val, field):
    try:
        result = [0] * len(encoding[field])
        result[encoding[field].index(val)] = 1
        return result
    except:
        print("One-hot encoding error (value %s; field %s)" % (val, field))
        sys.exit()


def onehot_encode(row, header, field):
    # Inserts one-hot encoding vector replacing the original value in a csv row,
    # and patches the header
    if field in header:
        idx = header.index(field)
        row = row[:idx] + encode(row[idx], field) + row[idx+1:]
        header = header[:idx] + [field + '_' + v for v in encoding[field]] + header[idx+1:]

    else:
        print("Can't one-hot encode %s field" % field)
        sys.exit()

    return row, header


def compute(input_csv, output_csv):
    with open(input_csv, 'r') as fi:
        reader = csv.reader(fi)
        header = next(reader)

        if not all(d in header for d in encoding.keys()):
            print("Wrong header. Expected the fields %s" % encoding.keys())
            sys.exit()

        # Cleaup MBIDs field
        if header.index('_recordingmbid_') != 0:
            print("Wrong header. Expected the 'recordingmbid' field as the first column")
            sys.exit()
        header[0] = 'recordingmbid'

        with open(output_csv, 'w') as fo:
            writer = csv.writer(fo)
            write_header = True

            for row in reader:
                mbid = clean_mbid(row[0])
                if mbid is None:
                    print("Wrong MBID: %s" % row[0])
                    sys.exit()
                row[0] = mbid

                header_new = header
                for field in encoding.keys():
                    row, header_new = onehot_encode(row, header_new, field)

                # Dump the new header to file the first time it occurs defined
                if write_header:
                    writer.writerow(header_new)
                    write_header = False

                writer.writerow(row)
    return


if __name__ == '__main__':
    parser = ArgumentParser(description = "Cleanup recording MBIDs and one-hot encode categorical data (%s)" % ' '.join(encoding.keys()))
    parser.add_argument("input_csv", help="input CSV feature matrix")
    parser.add_argument("output_csv", help="output CSV file")

    args = parser.parse_args()

    compute(args.input_csv, args.output_csv)

import sys
import csv
import json
import pandas
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, \
                            roc_auc_score
from argparse import ArgumentParser


def load_groundtruth(file, index=None):
    """
    Load ground truth from a CSV matrix. Reorder the rows according to index.
    :param file: CSV filename
    :param index: list of MBIDs to re-order recordings in the annotation
                  matrix. Follow the ascending order if None)
    :return: 2D binary numpy matrix [recordings, classes] with ground truth
             list of ground-truth classes corresponding to the matrix columns
             list of MBIDs corresponding to the matrix rows
    """
    gt = pandas.read_csv(file, index_col=0)
    if index:
        gt = gt.reindex(index)
    else:
        gt = gt.sort_index()
    gt_array = gt.values
    gt_classes = list(gt)
    gt_mbids = gt.index.tolist()
    print("Loaded ground-truth matrix (%d MBIDs, %d classes)" % gt_array.shape)
    return gt_array, gt_classes, gt_mbids


def load_predictions(file, index):
    """
    Load predictions from a NPY matrix file, and associated MBIDs from the
    index CSV file.
    :param file: NPY filename
    :param index: CSV index filename
    :return: 2D numpy matrix [recordings, classes] with predictions
             list of MBIDs corresponding to the matrix rows
    """
    pr_array = np.load(file)

    with open(index, 'r') as f:
        reader = csv.reader(f)
        pr_mbids = [x[0] for x in list(reader)]

    if len(pr_mbids) != pr_array.shape[0]:
        print("Predictions index size does not match the NPY array")
        sys.exit()

    print("Loaded predictions matrix (%d MBIDs, %d classes)" % pr_array.shape)
    return pr_array, pr_mbids


def compute_thresholds(predictions, groundtruth, classes):
    """
    Calculate several information retrieval (IR) metrics, like ROC AUC,
    precision, recall, f_score, etc.
    :param predictions: predictions, 2D numpy matrix (samples, classes)
    :param groundtruth: ground-truth, 2D binary numpy matrix (samples, classes)
    :param classes: list of class string names
    :return: dictionary of IR-metrics
    """

    # Optimized macro F-score
    macro_f_threshold = {}

    if len(classes) != predictions.shape[1]:
        print("Number of ground-truth classes does not match the predictions matrix")
        sys.exit()

    for i in range(len(classes)):
        precision, recall, threshold = precision_recall_curve(groundtruth[:, i], predictions[:, i])
        f_score = np.nan_to_num((2 * precision * recall) / (precision + recall))
        macro_f_threshold[classes[i]] = float(threshold[np.argmax(f_score)])

    """
    # A "micro-average": quantifying score on all classes jointly
    precision, recall, threshold = precision_recall_curve(groundtruth.ravel(), predictions.ravel())
    f_score = np.nan_to_num((2 * precision * recall) / (precision + recall))
    micro_f_threshold = float(threshold[np.argmax(f_score)])
    """

    return macro_f_threshold #, micro_f_threshold


def store_predictions_to_tsv(file, predictions, recordings, classes):
    """
    Store binary predictions to TSV file in MediaEval submission format
    :param file: TSV filename
    :param predictions: 2D binary matrix with predictions [recordings, classes]
    :param recordings: recording MBIDs corresponding to matrix rows
    :param classes: ground-truth classes corresponding to matrix columns 
    """

    if len(recordings) != predictions.shape[0]:
        print("MBID index does not match the predictions matrix")
    if len(classes) != predictions.shape[1]:
        print("List of ground-truth classes does not match the predictions matrix")

    print("Writing results to", file)
    with open(file, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for i in range(len(recordings)):
            predicted_classes = []
            for c in range(len(classes)):
                if predictions[i,c] == 1:
                    predicted_classes.append(classes[c])
            writer.writerow([recordings[i]] + predicted_classes)
    return


def compute(groundtruth_csv, predictions_npy, predictions_index, results_tsv, 
            test_predictions_npy=None, test_predictions_index=None, test_results_tsv=None):
    print("Validation")
    # Load predictions for validation
    pr_array, pr_mbids = load_predictions(predictions_npy, predictions_index)
    gt_array, gt_classes, gt_mbids = load_groundtruth(groundtruth_csv, index=pr_mbids)

    if pr_mbids != gt_mbids:
        print("Prediction MBID index does not match the ground truth")
        sys.exit()

    # Macro-averaged ROC AUC
    macro_f_threshold = compute_thresholds(pr_array, gt_array, gt_classes) #, micro_f_threshold 

    results = {}
    results['roc_auc'] = roc_auc_score(gt_array, pr_array)
    results['macro_f_threshold'] = macro_f_threshold
    #results['micro_f_threshold'] = micro_f_threshold

    # Dump thresholds to file
    with open(results_tsv + '.valid.json', 'w') as f:
        json.dump(results, f)

    for c in gt_classes:
        if macro_f_threshold[c] == 0:
            print("%s class is always predicted to maximize F-score" % c)

    # Predictions optimized for macro F-score
    pr_array_macro = np.copy(pr_array)
    for i in range(len(gt_classes)):
        threshold = macro_f_threshold[gt_classes[i]]
        pr_array_macro[:, i][pr_array_macro[:, i] < threshold] = 0
        pr_array_macro[:, i][pr_array_macro[:, i] >= threshold] = 1
    
    # Fix empty predictions for some recordings: 
    # predict the class with the highest activation relative to its threshold
    empty = np.where(~pr_array_macro.any(axis=1))[0]
    print("Thresholding results in %d empty recordings" % len(empty))
    for id in empty:
        # TODO store thresholds in a numpy array instead of a dictionary
        # Adapt threshold computation function to get rid of the list of classes
        # Store thresholds in a dict only when dumping to json 
        for i in range(len(gt_classes)):
            # empty predictions = no zero thresholds
            pr_array_macro[id, i] = pr_array[id, i] / macro_f_threshold[gt_classes[i]]
        max_i = np.argmax(pr_array_macro[id])
        pr_array_macro[id, :] = 0
        pr_array_macro[id, max_i] = 1

    store_predictions_to_tsv(results_tsv + '.max_f_macro', pr_array_macro, pr_mbids, gt_classes)

    """
    # Predictions optimized for micro F-score
    pr_array_micro = pr_array
    pr_array_micro[pr_array_micro < micro_f_threshold] = 0
    pr_array_micro[pr_array_micro >= micro_f_threshold] = 1
    store_predictions_to_tsv(results_tsv + '.max_f_micro', pr_array_micro, pr_mbids, gt_classes)
    """

    # Run predicions for test if configured:
    if test_predictions_npy and test_predictions_index and test_results_tsv:
        print("Test")
        test_pr_array, test_pr_mbids = load_predictions(test_predictions_npy, test_predictions_index)

        # Predictions on test set optimized for macro F-score
        test_pr_array_macro = np.copy(test_pr_array)
        for i in range(len(gt_classes)):
            threshold = macro_f_threshold[gt_classes[i]]
            test_pr_array_macro[:, i][test_pr_array_macro[:, i] < threshold] = 0
            test_pr_array_macro[:, i][test_pr_array_macro[:, i] >= threshold] = 1

        # Fix empty predictions (see comments above for TODO)
        empty = np.where(~test_pr_array_macro.any(axis=1))[0]
        print("Thresholding results in %d empty recordings" % len(empty))
        for id in empty:
            for i in range(len(gt_classes)):
                # empty predictions = no zero thresholds
                test_pr_array_macro[id, i] = test_pr_array[id, i] / macro_f_threshold[gt_classes[i]]
            max_i = np.argmax(test_pr_array_macro[id])
            test_pr_array_macro[id, :] = 0
            test_pr_array_macro[id, max_i] = 1

        store_predictions_to_tsv(test_results_tsv + '.max_f_macro', test_pr_array_macro, test_pr_mbids, gt_classes)

        """
        # Predictions on test optimized for micro F-score
        test_pr_array_micro = test_pr_array
        test_pr_array_micro[test_pr_array_micro < micro_f_threshold] = 0
        test_pr_array_micro[test_pr_array_micro >= micro_f_threshold] = 1
        store_predictions_to_tsv(test_results_tsv + '.max_f_micro', test_pr_array_micro, test_pr_mbids, test_gt_classes) 
        """

    print("All done.")


if __name__ == '__main__':
    parser = ArgumentParser(description="Tune on validation set and predict on test set")
    parser.add_argument('validation_groundtruth_csv', help="CSV file with a genre ground truth matrix")
    parser.add_argument('validation_predictions_npy', help="NPY matrix file with genre predictions matrix")
    parser.add_argument('validation_predictions_index', help="TSV file with a genre predictions index (MBIDs)")
    parser.add_argument('validation_results', help="filename prefix for validation results (TSVs with final predictions, JSON file with tuning parameters")
    parser.add_argument('test_predictions_npy', help="NPY matrix file with genre predictions matrix")
    parser.add_argument('test_predictions_index', help="TSV file with a genre predictions index (MBIDs)")
    parser.add_argument('test_results', help="TSVs with final predictions on test dataset")
    args = parser.parse_args()

    compute(args.validation_groundtruth_csv, args.validation_predictions_npy,
            args.validation_predictions_index, args.validation_results,
            args.test_predictions_npy, args.test_predictions_index,
            args.test_results)

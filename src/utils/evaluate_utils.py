import numpy as np

def kappa_cohen(ground_truth, predictions):
    ground_truth = ground_truth.view(-1).int().numpy()
    predictions = predictions.view(-1).int().numpy()
    tp = np.logical_and(ground_truth, predictions).sum()
    tn = np.logical_and(ground_truth, predictions).sum()
    fp = np.logical_and(ground_truth, predictions).sum()
    fn = np.logical_and(ground_truth, predictions).sum()
    total = (tp + tn + fp + fn)
    print(tp, tn, fp, fn)
    assert total is len(ground_truth) is len(predictions), "error in calculation of tp, tn, fp or fn"
    observed_agreement = (tp+tn) / total
    expected_yes = ((tp + fp) / total) + ((tp + fn) / total)
    expected_no = ((fn + tn) / total) + ((fp + tn) / total)
    expected_agreement = expected_yes + expected_no
    k = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return k

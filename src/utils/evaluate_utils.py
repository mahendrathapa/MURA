from src.config.config import Config


def kappa_cohen(ground_truth, predictions, verbose=False):
    ground_truth = ground_truth.view(-1) >= Config.CUT_OFF_THRESHOLD
    predictions = predictions.view(-1) >= Config.CUT_OFF_THRESHOLD

    ground_truth_c = ground_truth ^ 1
    predictions_c = predictions ^ 1

    tp = (ground_truth & predictions).sum().item()
    tn = (ground_truth_c & predictions_c).sum().item()
    fp = (ground_truth_c & predictions).sum().item()
    fn = (ground_truth & predictions_c).sum().item()
    total = (tp + tn + fp + fn)
    assert total==len(ground_truth)==len(predictions), "error in calculation of tp, tn, fp or fn"

    observed_agreement = (tp+tn) / total
    expected_yes = ((tp + fp) / total) * ((tp + fn) / total)
    expected_no = ((fn + tn) / total) * ((fp + tn) / total)
    expected_agreement = expected_yes + expected_no
    try:
        k = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    except ZeroDivisionError:
        k = 0.0
    if verbose:
        print("True Positive: {}".format(tp))
        print("False Positive: {}".format(fp))
        print("False Negative: {}".format(fn))
        print("True Negative: {}".format(tn))
        print("Observed Agreement: {}".format(observed_agreement))
        print("Expected Yes: {}".format(expected_yes))
        print("Expected No: {}".format(expected_no))
        print("Expected Agreement: {}".format(expected_agreement))
        print("Cohen Kappa Score: {}".format(k))
    return k


if __name__ == "__main__":
    import torch
    gt_1 = torch.tensor([0, 1, 0, 1])
    p_1 = torch.tensor([1, 1, 1, 0])
    # for gt_1 & p_1 k = -0.5
    print(kappa_cohen(gt_1, p_1, verbose=True))
    gt_2 = torch.tensor([0])
    p_2 = torch.tensor([1])
    print(kappa_cohen(gt_2, p_2, verbose=True))

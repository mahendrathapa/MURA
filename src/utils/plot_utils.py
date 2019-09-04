from matplotlib import pyplot as plt


def generate_plot(iteration, train_result, val_result, title=None, y_label=None):

    fig = plt.figure(figsize=(15, 8))
    plt.plot(iteration, train_result, label="train")
    plt.plot(iteration, val_result, label="val")
    plt.title(title)
    plt.xlabel("epochs")
    plt.ylabel(y_label)
    plt.legend(loc='upper right')

    return fig

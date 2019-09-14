import numpy as np
import torch

from copy import deepcopy
from matplotlib import cm

from src.utils.data_utils import get_image
from src.architecture.densenet import Densenet
from src.constants import Constants


def load_network(checkpoint_path, device='cpu'):
    print("Loading checkpoint {}".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location=device)

    network = Densenet()
    network.load_state_dict(checkpoint["network_dict"])

    return network


def predict_with_cam(network, to_predict, final_conv_layer='conv',
                     fc_layer='fc', device='cpu'):

    features = list()

    def get_features(module, input, output):
        features.extend((output.cpu().data).numpy())

    image_original = get_image(to_predict, unsqueeze_dim=2)
    image = deepcopy(image_original)
    image_original = image_original.squeeze(0).squeeze(0).cpu().numpy()

    layer_to_hook = network._modules.get(final_conv_layer)
    hook = layer_to_hook.register_forward_hook(get_features)
    with torch.no_grad():
        image = image.to(device)
        prediction = network(image)
        prediction_probability = torch.sigmoid(prediction)
    hook.remove()

    image = image.squeeze(0).squeeze(0).cpu().numpy()

    feature = features[0]
    nc, h, w = feature.shape
    weight_softmax_params = list(
        network._modules.get(fc_layer).parameters()
    )
    weight_softmax = np.squeeze(
        weight_softmax_params[0].cpu().data.numpy()
    )

    cam = weight_softmax.dot(feature.reshape((nc, h * w)))
    cam = cam.reshape((h, w))
    cam -= np.mean(cam)
    cam /= np.max(cam)

    cam = torch.from_numpy(cam).float()
    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = torch.nn.functional.interpolate(
        cam, size=image.shape, mode='bilinear', align_corners=True
    )
    cam = cam.squeeze(0).squeeze(0).cpu().numpy()

    colormap = cm.get_cmap('jet')
    heatmap = colormap(cam)
    heatmap = np.uint8(heatmap[:, :, :3] * 255)
    image_heatmap = 0.2 * heatmap + 0.8 * np.expand_dims(image_original, -1)
    image_heatmap = np.uint8(image_heatmap)

    if prediction_probability.item() >= Constants.DECISION_PROBABILITY:
        prediction_class = Constants.POSITIVE_CLASS
    else:
        prediction_class = Constants.NEGATIVE_CLASS

    result = {
        "image": image_original,
        "label": prediction_class,
        "prediction_probability": prediction_probability.item(),
        "heatmap": image_heatmap
    }
    return result

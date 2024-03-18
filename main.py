import cv2
import numpy as np
import itertools

from core.filter import GuidedFilter
from tools import visualize as vis
from cv.image import to_8U, to_32F
import matplotlib.pyplot as plt

cmap = plt.cm.turbo

def data_colorize(data, max=10.0, min=0.0, vis_max=10.0, vis_min=0.0, inv=True, zero_mask=False): 
    """
    Colorize the given data according to the specified range.

    Args:
    data (np.array): Input data to be colorized [0-1].
    max (float): Maximum value of the input data.
    min (float): Minimum value of the input data.
    vis_max (float): Maximum value to be visualized.
    vis_min (float): Minimum value to be visualized.
    inv (bool): Whether to invert the data or not.
    zero_mask (bool): Whether to mask zero values or not.

    Returns:
    np.array: Colorized data as a uint8 array.
    """

    # Create a mask for non-zero elements
    mask = data != 0
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Restore the original data range
    data = data * (max - min) + min
    
    # Normalize to new range
    data = (data - vis_min) / (vis_max - vis_min)

    # Invert the data if inv is True
    if inv: 
        data = 1 - data

    # Clip the data to the range [0, 1]
    data = np.clip(data, 0, 1)

    # Apply colormap and scale the values to the range [0, 255]
    data = 255.0 * cmap(data)[:, :, :3]

    # Mask zero values if zero_mask is True
    if zero_mask: 
        data = data * mask

    return data.astype('uint8')

def save_phase_as_uint8colored(img, filename, zero_mask=False):
    #from tensor
    img = data_colorize(img, max=40, min=-40, vis_max=5.0, vis_min=-8.0, inv=True)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

def test_gray():
    image = cv2.imread('data/cat.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    radius = [2, 4, 8]
    eps = [0.1**2, 0.2**2, 0.4**2]

    combs = list(itertools.product(radius, eps))

    vis.plot_single(image, title='origin')
    for r, e in combs:
        GF = GuidedFilter(image, radius=r, eps=e)
        vis.plot_single(GF.filter(image), title='r=%d, eps=%.2f' % (r, e))


def test_color():
    if 0:
        image = cv2.imread('data/Lenna.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        noise = (np.random.rand(image.shape[0], image.shape[1], 3) - 0.5) * 50
        image_noise = image + noise
    else:
        image = cv2.imread('data/00095_rgb.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_noise = cv2.imread('data/00095_ppred_mono.png')

    if 1:
        image = cv2.resize(image, None, fx=0.5, fy=0.5)
        image_noise = cv2.resize(image_noise, None, fx=0.5, fy=0.5)


    # radius = [1, 2, 4]
    radius = [20]
    # eps = [0.005]
    eps = [0.0001]
    itr_num = 5

    combs = list(itertools.product(radius, eps))

    # vis.plot_single(to_32F(image), title='origin')
    # vis.plot_single(to_32F(image_noise), title='noise')

    for r, e in combs:
        GF = GuidedFilter(image, radius=r, eps=e)
        image_filtered = image_noise
        for i in range(itr_num):
            print('iteration:', i)
            image_filtered = GF.filter(image_filtered)
        image_filtered = cv2.resize(image_filtered, None, fx=2.0, fy=2.0)
        image_filtered = cv2.cvtColor(image_filtered, cv2.COLOR_RGB2GRAY)
        cv2.imwrite('data/00095_ppred_mono_fil.png', (image_filtered*65535).astype('uint16'))
        save_phase_as_uint8colored(image_filtered, f'data/00095_ppred_fil_r{radius}_e{eps}.png')
        # vis.plot_single(to_32F(image_filtered), title='r=%d, eps=%.3f' % (r, e))


if __name__ == '__main__':
    # test_gray()
    test_color()

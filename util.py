import numpy as np
import cv2
import pdb


def cxy_wh_2_rect1(pos, sz):
    """
    :param pos: center coordinate of box, tuple (cx, cy)
    :param sz: width and hight of the box, tuple (w, h)
    :return: ndarray, [x1, y1, w, h], where x1 and y1 are the coordinate of the top left point
    """
    return np.array([pos[0]-sz[0]/2+1, pos[1]-sz[1]/2+1, sz[0], sz[1]])  # 1-index


def rect1_2_cxy_wh(rect):
    """
    :param rect: top left and bottom right coordinate of the box. (x1, y1, x2, y2)
    :return: ndarray, [x1, y1, w, h], where x1 and y1 are the coordinate of the top left point
    """
    return np.array([rect[0]+rect[2]/2-1, rect[1]+rect[3]/2-1]), np.array([rect[2], rect[3]])  # 0-index


def cxy_wh_2_bbox(cxy, wh):
    """
    :param cxy: center coordinate of the box, tuple (cx, cy)
    :param wh: width and hight of the box, tuple (w, h)
    :return: top left and bottom right coordinate of the box. (x1, y1, x2, y2)
    """
    return np.array([cxy[0]-wh[0]/2, cxy[1]-wh[1]/2, cxy[0]+wh[0]/2, cxy[1]+wh[1]/2])  # 0-index


def gaussian_shaped_labels(sigma, sz):
    """
    :param sigma: variance of the 2D gaussian distribution.
    :param sz: size of the gaussian kernel
    :return:
    """
    x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1]+1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g


def crop_chw(image, bbox, out_sz, padding=(0, 0, 0)):
    """ crop the box inside image and reshape to the out_sz
    :param image: input image. Ndarray, Shape (h, w, 3)
    :param bbox: the box cropped over the input image. Ndarray [x1, x2, y1, y2]
    :param out_sz: output image size. Int
    :param padding: padding
    :return: cropped image. Ndarray, Shape (3, out_sz, out_sz)
    """
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    # crop: (out_sz, out_sz, 3)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return np.transpose(crop, (2, 0, 1))


if __name__ == '__main__':
    a = gaussian_shaped_labels(10, [5,5])
    print(a)
import argparse
from skimage.io import imread, imsave

from inpainter import Inpainter
import numpy as np

def main():
    args = parse_args()

    # image shape: (r, c, 3)
    # image values: 0 - 255
    # mask shape: (r, c)
    # mask values: 1.0(parts to be filled in) or 0.0(parts not to be filled in)
    if '.npy' in args.input_image:
        image = np.load(args.input_image)
        mask = np.load(args.mask)
    else:
        image = imread(args.input_image)
        mask = imread(args.mask, as_gray=True)

    output_image = Inpainter(
        image,
        mask,
        patch_size=args.patch_size,
        plot_progress=args.plot_progress
    ).inpaint()
    imsave(args.output, output_image, quality=100)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-ps',
        '--patch-size',
        help='the size of the patches',
        type=int,
        default=9
    )
    parser.add_argument(
        '-o',
        '--output',
        help='the file path to save the output image',
        default='output.jpg'
    )
    parser.add_argument(
        '--plot-progress',
        help='plot each generated image',
        action='store_true',
        default=False
    )
    parser.add_argument(
        'input_image',
        help='the image containing objects to be removed'
    )
    parser.add_argument(
        'mask',
        help='the mask of the region to be removed'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()

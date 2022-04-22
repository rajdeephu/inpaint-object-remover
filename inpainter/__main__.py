import argparse
from skimage.io import imread, imsave

from inpainter import Inpainter
import numpy as np

def main():
    args = parse_args()

    # image shape: (r, c, *) or (b, r, c, *) with * as 3 or 1
    # image values: 0 - 255
    # mask shape: (r, c) or (b, r, c)
    # mask values: 1.0(parts to be filled in) or 0.0(parts not to be filled in)
    if '.npy' in args.input_image:
        image = np.load(args.input_image)
        mask = np.load(args.mask)
    else:
        image = imread(args.input_image)
        mask = imread(args.mask, as_gray=True)

    if args.grayscale:
        image = np.repeat(image, 3, -1)

    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

    for i in range(image.shape[0]):
        output_image = Inpainter(
            image[i],
            mask[i],
            patch_size=args.patch_size,
            plot_progress=args.plot_progress
        ).inpaint()

        if args.grayscale:
            output_image = output_image[:,:,0]
        
        imsave(args.output + str(i) + '.jpg', output_image, quality=100)

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
        default='output/image'
    )
    parser.add_argument(
        '-pp',
        '--plot-progress',
        help='plot each generated image',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '-gs',
        '--grayscale',
        help='input image is grayscale',
        action='store_true'
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

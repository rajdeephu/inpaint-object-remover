from inpainter import Inpainter
import numpy as np

# image shape: (b, r, c, *) with * as 3 or 1
# image values: 0 - 255
# mask shape: (b, r, c)
# mask values: 1.0(parts to be filled in) or 0.0(parts not to be filled in)
def get_inpainted(image, mask, patch_size=3, plot_progress=False, grayscale=False):
    if grayscale:
        image = np.repeat(image, 3, -1)

    image_list = []
    for i in range(image.shape[0]):
        output_image = Inpainter(
            image[i],
            mask[i],
            patch_size=patch_size,
            plot_progress=plot_progress
        ).inpaint()
        if grayscale:
            output_image = output_image[:,:,0]
        image_list.append(output_image)

    image_list = np.array(image_list)
    return image_list
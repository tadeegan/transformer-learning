import numpy as np

def tile_mnist(imgs, shape=(4,4)):
    # B, C, H, W
    width = int(imgs.shape[-2] / shape[1])
    height = int(imgs.shape[-1] / shape[0])
    patches = []
    patch_idx = 0
    for x in range(shape[1]):
        for y in range(shape[0]):
#             patch = imgs[..., 0]
            patch_idx += 1
            patch = imgs[..., y*height:y*height+height, x*width:x*width+width]
            patches.append(patch)
    return np.stack(patches, axis=1)

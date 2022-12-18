"""

 CV_IO_utils.py  (author: Anson Wong / git: ankonzoid)

"""
import os
import skimage.io
from multiprocessing import Pool

# Read image
def read_img(filePath):
    return skimage.io.imread(filePath, as_gray=False)

# Read images with common extensions from a directory
def read_imgs_dir(dirPath, extensions, parallel=True):
    print("HUH1")
    args = [os.path.join(dirPath, filename)
            for filename in os.listdir(dirPath)
            if any(filename.lower().endswith(ext) for ext in extensions)]
    print("HUH2")
    if parallel:
        print("HUHHU")
        pool = Pool()
        print("break here 1")
        print(args)
        imgs = pool.map(read_img, args)
        print("break here 2")
        pool.close()
        print("break here 3")
        pool.join()
        print("HUH3")
    else:
        print("UHHH")
        imgs = [read_img(arg) for arg in args]
    print("HUH4")
    return imgs

# Save image to file
def save_img(filePath, img):
    skimage.io.imsave(filePath, img)
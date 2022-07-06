from PIL import Image
import os
import numpy as np

def get_imlist(path):
	# Returns a list of filenames for all jpg images in a directory.
 	return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

images = get_imlist('dataset')
for f in range(len(images)):    
    im = Image.open(images[f])
    out = im.resize((40, 30))

    outfile = os.path.splitext(images[f])[0] + ".jpg"
    out.save(outfile)
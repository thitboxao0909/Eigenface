from PIL import Image
import os
import numpy as np

def get_imlist(path):
	# Returns a list of filenames for all jpg images in a directory.
 	return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

preprocess_image = get_imlist('preprocess')

for f in range(len(preprocess_image)):    
    im = Image.open(preprocess_image[f])
    #out = im.resize((224, 299))
    out = im.convert('L')
    out = out.resize((300, 300))
    out.show()
    outfile = os.path.splitext(preprocess_image[f])[0] + "_processed.jpg"
    out.save(outfile)
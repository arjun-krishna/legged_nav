"""
Generate GIF from images
"""
import glob
from PIL import Image
import argparse
import os

parser = argparse.ArgumentParser(description='Generate GIFs from camera capture frames')
parser.add_argument('-e', '--experiment', help='experiment/run name')

args = parser.parse_args()
os.chdir((os.path.dirname(os.path.abspath(__file__))))
imgs_dir = 'logs/%s/exported/frames/' % args.experiment    # viewer frames
outs_dir = 'logs/%s/exported/' % args.experiment           # output dir for GIF

# filepaths
fp_in = imgs_dir + "*.png"
fp_out = outs_dir + "viewer.gif"

def get_frame(s):
    return int(s.split('/')[-1].split('.')[0])

files = glob.glob(fp_in)
if len(files):
    files.sort(key=get_frame)
    img, *imgs = [Image.open(f) for f in files]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=17, loop=0)
    print('**Saved', fp_out)

import glob
import scipy.io as sio
import cv2
import os

if not os.path.exists('mat'):
	os.makedirs('mat')
l = glob.glob('*.png')
for i in l:
	img = cv2.imread(i, 0)
	img = img/255
	p = {}
	p['pred'] = img
	sio.savemat(os.path.join('mat',i.replace('png','mat')), p)



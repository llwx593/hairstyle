# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 23:51:35 2019

@author: 15440
"""

import os
import sys
from skimage import io

path = '1/'
topath = '2/'
count=1
if not os.path.exists(topath):
    os.mkdir(topath)
if len(sys.argv)==2:
	count=int(sys.argv[1])
for file in os.listdir(path):
    os.rename(os.path.join(path,file),os.path.join(topath,str(count)+".jpg"))
    count+=1
print("rename successfully!\nEnd with "+str(count-1))

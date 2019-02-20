import os 
import glob 

files = sorted(glob.glob('*.zip'))
for file in files:
    os.system('unzip -n {} -d ./'.format(file))


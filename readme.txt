Description:
Series of Python scripts as GH definitions allowing Graph Machine Learning within the Rhino|Grasshopper ecosystem.


########################################################
Setup for using Python inside Rhino 8 and GH:

1. Install python 3.9.10 to operating system
https://www.python.org/downloads/release/python-3910/

2. Go to GH environment folder using the Windows Command Prompt (type cdm in search box)
cd C:\Users\l03063058\.rhinocode\py39-rh8\Scripts
cd C:\Users\ROG Zephyrus\.rhinocode\py39-rh8\Scripts

3. Install library packages from here using command prompt in Windows
ex. pip install matplotlib

4. Open RH and GH
Use the Python 3 component
Double-click to insert the following code as a basis

#! python3
#r: openpyxl, psycopg2, Ipython
 
import pandas as pd
import matplotlib
 
######[INSERT CODE AFTER THIS LINE]#############################

 
Tips:
Only install library packages through command prompt inside GH environment folder (as in step 3)
Do not install library packages through package installer inside the python script editor within GH
If getting CPU error (or any other error), delete all the contents from C:\Users\l03063058\.rhinocode and restart Grasshopper
Sometimes Panda fails. Uninstall and re-install through command prompt inside GH environment folder
Use the language setting set to English to void some issues




Useful commands:
pip list
pip install package_name
pip uninstall package_name
pip install package_name==version
pip install --upgrade package_name

Needed libraries:
dgl 2.0.0
geopandas 1.0.1
matplotlib 3.9.0
networkx 3.2.1
numpy 2.0.2
osmnx 1.9.3
pandas 2.2.2
scikit-learn 1.5.1
seaborn 0.13.2
shapely 2.0.4
torch 2.2.1
torchdata 0.8.0
tqdm 4.66.5



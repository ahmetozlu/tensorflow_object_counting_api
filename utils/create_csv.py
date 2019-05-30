#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 21st September 2017
#----------------------------------------------

import sys
import os.path
import csv

# This is a tiny script to help you creating a CSV file from a face database with a similar hierarchie:
#  .
#  |-- README
#  |-- s1
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  |-- s2
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  ...
#  |-- s40
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#

def CreateCsv(csvLine):  
    
    BASE_PATH=csvLine
    SEPARATOR=","

    label = 0
    with open("face_dataset.csv", "w") as file:
	    writer = csv.writer(file)
	    for dirname, dirnames, filenames in os.walk(BASE_PATH):
		for subdirname in dirnames:
		    subject_path = os.path.join(dirname, subdirname)
		    for filename in os.listdir(subject_path):
		        abs_path = "%s/%s" % (subject_path, filename)
		        csv_line = "%s%s%d" % (abs_path, SEPARATOR, label)
			print csv_line			    		
	    		writer.writerows([csv_line.split(',')])
		    label = label + 1


import os
my_list = os.listdir("./")
import fileinput

for the_file in my_list:
	text_to_search = "test_images"
	replacement_text = "test"

	if (".xml" in the_file):
		temp = ""
		with open(the_file) as fp:  
			my_line = fp.readline()
			cnt = 1
			while my_line:
				my_line = fp.readline()
				temp2 = my_line
				if "filename" in temp2:
					temp = temp2.replace('<filename>','')
					temp = temp.replace('</filename>','')
				cnt += 1

		with fileinput.FileInput(the_file, inplace=True, backup='.bak') as file:
			for line in file:
			    print(line.replace(text_to_search, replacement_text + "/" + temp.strip()), end='')

my_list = os.listdir("./")
for the_file in my_list:
	if(".bak" in the_file):
		os.remove(the_file)


import csv

with open('adult.data') as input_file:
    lines = input_file.readlines()
    newLines = []
    for line in lines:
        newLine = line.strip().split()
        newLines.append( newLine )

with open('adult.csv', 'wb') as test_file:
    file_writer = csv.writer(test_file)
    file_writer.writerows( newLines )
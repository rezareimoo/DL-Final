import os

# abspath to a folder as a string
folder = 'data/ModelNet40'

# GET ALL INCORRECT FILES AND STORE IN TXT:

# files_ = []
# first_lines = []
# count = 0
# for dirname, dirs, files in os.walk(folder):
#     for filename in files:
#         count += 1
#         f_path= os.path.join(dirname, filename)
#         with open(f_path, 'r') as f:
#             line = f.readline()
#             if line.strip() != 'OFF':
#                 files_.append(f_path + '\n')
#             f.close()
# print(count)
# print(len(files_))
# with open('incorrect_files.txt', 'w+') as f:
#     f.writelines(files_)    
     



# SCRIPT FOR FIXING THE FILES:

with open('incorrect_files.txt', 'r') as f:
    file_paths = f.readlines()
    for filename in file_paths:
        with open(filename.strip(), 'r') as f:
            lines = f.readlines()
            f.close()
            if lines[0].strip() != 'OFF':
                with open(filename.strip(), 'w') as f:
                    x = lines[0][3:]
                    lines[0] = x
                    lines.insert(0, 'OFF\n')
                    f.writelines(lines)
                    f.close()

for dirname, dirs, files in os.walk(folder):
    for filename in files:
        with open(os.path.join(dirname, filename), 'r') as f:
            line = f.readline()
            if line.strip() != 'OFF':
                print('didn\'t work')
                quit()
            f.close()

print('done')
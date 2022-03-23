import sys

file_path = sys.argv[1]

fid_score = []

# readline_all.py
f = open(file_path, 'r')
while True:
    line = f.readline()
    if "FID SCORE:" in line:
        fid_score.append(float(line.split("FID SCORE: ")[-1]))
    if not line: break
    # print(line) 
f.close()

for i, fid in enumerate(fid_score):
    print(fid,end=",")
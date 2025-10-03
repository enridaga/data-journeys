import json,sys

def extractSourceCode(jj):
    src=""
    for cell in jj['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if(isinstance(source, str) ):
                src = src + "\n" + source
            else:
                src = src + "\n"
                for line in source:
                    src = src + line
    return src

print("# Notebook", sys.argv[1])
with open(sys.argv[1], "r") as f:
    j = json.load(f)
    src = extractSourceCode(j)
    print(src)
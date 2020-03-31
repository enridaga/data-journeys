#!/usr/local/bin/python3.7
import datajourney as DJ
import json as J
from os import listdir
from os.path import isfile, join

simple_src = """
x = 1
y = "2"
r = x + y
g = prepare(r)
visualize(g, x, y)

#a = 10

def visualize(a, b ,c):
  print(a, b, c)

bag = []
for q in g:
 bag.append(q)

bag = bag.clone()

#df_over_100mb = df[df['max_filesize'] >= 100000000]
#df.rename(columns = {'Unnamed: 0':'nb_id'}, inplace = True)
#print(df_over_100kb['max_filesize'].sum() / 1000000000)

bag.autoUpdate()

while x < 10:
  x += 1
  y -= 1
  c *= 2

obj.doIt(bag)

"""

def extractSourceCode(notebook):
    src=""
    for cell in jj['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if(isinstance(source, str) ):
                src = src + "\n" + source
            else:
                for line in source:
                    src = src + "\n" + line
    return src

import re

indir = "kernels/"
srcdir = "sources/"
outdir = "graphs/"
files = [f for f in listdir(indir) if isfile(join(indir, f))]
for f in files:
    with open(indir + f) as notebook:
        print("Processing: {0}".format(f))
        jj = J.load(notebook)
        src = extractSourceCode(jj)
        #
        comm = ["^%","\n%","^!","\n!"]
        for c in comm:
            src = re.sub(c, "\n### ", src)

        s = open(srcdir + f[:-5] + "py", "w")
        s.write(src)
        collector = DJ.FindDependencies()
        try:
            collector.collect(src)
            g = collector.getStringCollected()
            # Save
            o = open(outdir + f[:-5] + "digraph", "w")
            o.write(g)
        except AttributeError as err:
            print("AttributeError: {0} [{1}]".format(err, f))
        except SyntaxError as err:
            print("SyntaxError: {0} [{1}]".format(err, f))
        except TypeError as err:
            print("TypeError: {0} [{1}]".format(err, f))



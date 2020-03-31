#!/usr/local/bin/python3.7
import datajourney as DJ
import json as J
from os import listdir
from os.path import isfile, join
from graphviz import Source

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
imgdir = "images/"
files = [f for f in listdir(indir) if isfile(join(indir, f))]
for f in files:
    with open(indir + f) as notebook:
        print("Processing: {0}".format(f))
        jj = J.load(notebook)
        src = extractSourceCode(jj)
        if src.strip() == "":
            print("EMPTY: no source code in notebook {0}".format(f))
            continue
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
            # src = Source(g)
            # src.format = "png"
            # src.render(imgdir + f[:-5])
        except AttributeError as err:
            print("AttributeError: {0} [{1}]".format(err, f))
        except SyntaxError as err:
            print("SyntaxError: {0} [{1}]".format(err, f))
        except TypeError as err:
            print("TypeError: {0} [{1}]".format(err, f))
        except Exception as err:
            print("Exception: {0} [{1}]".format(err, f))



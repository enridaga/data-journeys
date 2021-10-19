#!/usr/local/bin/python3.7
import sys
import json as J
from os import listdir
from os.path import isfile, join
from graphviz import Source

srcdir = "sources/"
outdir = "graphs/"
imgdir = "images/"

nb = sys.argv[1]

# src = Source(g)
# src.format = "png"
# src.render(imgdir + f[:-5])
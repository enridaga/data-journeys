#!/usr/local/bin/python3.7
import datajourney as DJ
import json as J
from os import listdir
from os.path import isfile, join
from graphviz import Source
import pygraphviz
import networkx.drawing, networkx.drawing.nx_agraph as ag

outdir=""
imgdir="images/"
indir="sources/"
f="2019-data-science-bowl-via-autoviml.py"
f="ultra-fast-qwk-calc-method.py" # AttributeError: 'Subscript' object has no attribute 'id' 
f="time-travel-eda.py" # AttributeError: 'Subscript' object has no attribute 'id' 
f="ieee-internal-blend.py" # AttributeError: 'Subscript' object has no attribute 'id' 
f="guide-to-animated-bubble-charts-using-plotly.py"
f="top-5-data-visualization-libraries-tutorial.py"
f="starter-explore-youtube8m-sample-data.py"
f="eda-script-67.py"
f="practical-model-evaluation-day-1.py" # AttributeError: 'FindDependencies' object has no attribute '_bag'
f="how-to-attack-a-machine-learning-model.py" # AttributeError: 'Attribute' object has no attribute 'id'
f="very-simple-pytorch-training-0-59.py" # CLASS support - still to finish
# f="keras-u-net-starter-lb-0-277.py" # TODO AttributeError: 'Call' object has no attribute 'id'
src = open(indir + f, "r").read()
# print(src)
collector = DJ.FindDependencies(f[:-3])
collector.collect(src)
s = collector.getStringCollected()
# # Save
o = open(outdir + f[:-2] + "digraph", "w")
o.write(s)
# #
# src = Source(g)
# src.format = "png"
# src.render(imgdir + f[:-3])

g = ag.read_dot(outdir + f[:-2] + "digraph")

for edge in g.edges.data('label'):
    print(edge)
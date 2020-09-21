#!/usr/local/bin/python3.7
# TO RDF
import datajourney as DJ
import base64
import json as J
from os import listdir
from os.path import isfile, join
from graphviz import Source
import pygraphviz
from rdflib import URIRef, BNode, Literal, Namespace, Graph
from rdflib.namespace import CSVW, DC, DCAT, DCTERMS, DOAP, FOAF, ODRL2, ORG, OWL, \
                           PROF, PROV, RDF, RDFS, SDO, SH, SKOS, SOSA, SSN, TIME, \
                           VOID, XMLNS, XSD
import networkx.drawing, networkx.drawing.nx_agraph as ag

outdir="graphs/"
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
f="0-cpe-getting-familier-with-problem-and-dataset.py"
#f="keras-u-net-starter-lb-0-277.py" # TODO AttributeError: 'Call' object has no attribute 'id'
# src = open(indir + f, "r").read()
# # print(src)
# collector = DJ.FindDependencies(f[:-3])
# collector.collect(src)
# s = collector.getStringCollected()
# # Save
# o = open(outdir + f[:-2] + "digraph", "w")
# o.write(s)
# # Save
# #
g = ag.read_dot(outdir + f[:-2] + "digraph")

######
n = f[:-3]

rdfg = DJ.toRDF(n, g)
print(rdfg.serialize(format="turtle").decode("utf-8"))
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
#f="keras-u-net-starter-lb-0-277.py" # TODO AttributeError: 'Call' object has no attribute 'id'
src = open(indir + f, "r").read()
# print(src)
collector = DJ.FindDependencies(f[:-3])
collector.collect(src)
s = collector.getStringCollected()
# Save
o = open(outdir + f[:-2] + "digraph", "w")
o.write(s)
# # Save
# #
g = ag.read_dot(outdir + f[:-2] + "digraph")

######
n = f[:-3]

# def toRDF(name, digraph):
#     n = name
#     g = digraph
#     DJ = Namespace("http://purl.org/dj/")
#     K = Namespace("http://purl.org/dj/kaggle/")
#     L = Namespace("http://purl.org/dj/python/lib/")
#     notebook = URIRef(str(K) + n)
#     Loc = Namespace(str(K) + str(n) + "#")
#     #print(notebook)
#     rdfg = Graph()
#     rdfg.bind("rdf", RDF)
#     rdfg.bind("dj", DJ)
#     rdfg.bind("rdfs", RDFS)
#     rdfg.bind("k", K)
#     rdfg.add((getattr(K, notebook), RDF.type, getattr(K, "Notebook")))
#     for edge in g.edges.data('label'):
#         pl = edge[2]
#         sl = edge[0]
#         ol = edge[1]
#         # If predicate Imports, use LIB namespace on Subject
#         if pl == "import":
#             subj = URIRef(str(L) +str(hash(sl)))
#         else:
#             subj = URIRef(str(Loc) + str(hash(sl)))
#         # If object is notebook, use Notebook entity instead
#         if ol == n:
#             obj = notebook
#         else:
#             obj = URIRef(str(Loc) + str(hash(ol)))
#         pred = URIRef(str(DJ) + pl)
#         rdfg.add((subj, RDFS.label, Literal(sl)))
#         rdfg.add((obj, RDFS.label, Literal(ol)))
#         rdfg.add((subj, pred, obj))
#     return rdfg
    

rdfg = DJ.toRDF(n, g)
print(rdfg.serialize(format="turtle").decode("utf-8"))
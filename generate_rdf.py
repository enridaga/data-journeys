
import datajourney as DJ
import json as J
from os import listdir
from os.path import isfile, join
from graphviz import Source
import networkx.drawing, networkx.drawing.nx_agraph as ag


indir = "graphs/"
# srcdir = "sources/"
outdir = "rdf/"

files = [f for f in listdir(indir) if isfile(join(indir, f))]
for f in files:
    with open(indir + f) as notebook:
        print("Processing: {0}".format(f))
        try:
            g = ag.read_dot(indir + f)
            n = f[:-7]
            rdfg = DJ.toRDF(n[:-1], g)
            print("Writing to: {0}".format(outdir + f[:-7] + "ttl"))
            rdfg.serialize(destination=outdir + f[:-7] + "ttl", format="ttl")
        except Exception as err:
            print("ERROR Exception: {0} [{1}]".format(err, f))
            #print("ERROR Some error occurred with: {0}".format(f))
            
        # try:
        #     # Save
        #
        #     rdfg.write(outdir + f[:-5] + "ttl", format="nt")
        #
        # except AttributeError as err:
        #     print("AttributeError: {0} [{1}]".format(err, f))
        # except SyntaxError as err:
        #     print("SyntaxError: {0} [{1}]".format(err, f))
        # except TypeError as err:
        #     print("TypeError: {0} [{1}]".format(err, f))
        # except Exception as err:
        #     print("Exception: {0} [{1}]".format(err, f))



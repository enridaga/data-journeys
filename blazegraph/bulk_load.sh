#!/bin/bash
# Assumes the file blazegraph.jar is in this directory!
# https://github.com/blazegraph/database/releases
java -cp *:*.jar com.bigdata.rdf.store.DataLoader -namespace dj -defaultGraph http://purl.org/dj/data/ -verbose blazegraph.properties ../rdf/

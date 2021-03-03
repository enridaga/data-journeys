#!/bin/bash
source dj-py3.9/bin/activate

counter=0
limit=100
for f in $(ls -Sr rdf)
do
    counter=$((counter+1))
    if (( counter > limit )); then
        exit 0
    fi
    fn="${f%.*}"
    echo "Processing $fn"
    papermill DataJourneyGenerator.ipynb /dev/null -p notebook "$fn"
done
#!/bin/bash
#source dj-py3.9/bin/activate
source dj-conda-py3.8/bin/activate

counter=0
limit=1000
for f in $(ls -Sr rdf)
do	
    counter=$((counter+1))
    if (( counter > limit )); then
        exit 0
    fi
    fn="${f%.*}"
    echo "Processing $fn"
    if [ "$fn" = "exploratory-image-analysis" ]; then
		continue
	fi
    if [ "$fn" = "andrews-script-plus-a-genetic-program-model" ]; then
		continue
	fi
    if [ "$fn" = "more-text-cleaning-to-increase-word-coverage" ]; then 
		# The message was: Cell execution timed out
		continue
	fi
    if [ "$fn" = "how-to-attack-a-machine-learnttl" ]; then 
		# The message was: Cell execution timed out
		continue
	fi
	# ashrae-great-energy-predictor-iii-eda-model
	# eloda-with-feature-engineering-and-stackin
	target=datajourneys/$fn
	if test -f "$target"; then
		# if file exists, skip
		echo "Skipping $fn"
	else
		echo "Doing $fn" # --execution-timeout 1200 --autosave-cell-every 0
    	papermill DataJourneyGenerator.ipynb "DJG_Log-$fn.ipynb" -p notebook "$fn"
	fi
done

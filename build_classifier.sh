#!/bin/bash
source dj-py3.9/bin/activate
papermill MultiClassificationExperiments.ipynb "./experiments_output/MultiClassificationExperiments_rdf2vec_r2_s1000.ipynb" -p emb_method rdf2vec -p test_regime 2 -p input_size 1000 -p output_file build_classifier.csv
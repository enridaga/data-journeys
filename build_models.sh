#!/bin/bash
source dj-py3.9/bin/activate

# papermill MultiClassificationExperiments.ipynb "./models/MultiClassificationExperiments_rdf2vec_r1_s20.ipynb" -p emb_method rdf2vec -p test_regime 2 -p input_size 20 -p output_file MultiClassificationExperiments.csv -p save_models True

# exit 1
# 
papermill MultiClassificationExperiments.ipynb "./models/MultiClassificationExperiments_rdf2vec_r1_s200.ipynb" -p emb_method rdf2vec -p test_regime 1 -p input_size 200 -p output_file MultiClassificationExperiments.csv -p save_models True
#
papermill MultiClassificationExperiments.ipynb "./models/MultiClassificationExperiments_rdf2vec_r2_s200.ipynb" -p emb_method rdf2vec -p test_regime 2 -p input_size 200 -p output_file MultiClassificationExperiments.csv -p save_models True
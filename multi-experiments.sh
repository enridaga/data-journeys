#!/bin/bash
source dj-py3.8/bin/activate
# 
# 10
for i in {1..10}; do papermill MultiClassificationExperiments.ipynb "./experiments_output/MultiClassificationExperiments_rdf2vec_r1_s10_i$i.ipynb" -p emb_method rdf2vec -p test_regime 1 -p input_size 10 -p output_file MultiClassificationExperiments.csv; done
for i in {1..10}; do papermill MultiClassificationExperiments.ipynb "./experiments_output/MultiClassificationExperiments_rdf2vec_r2_s10_i$i.ipynb" -p emb_method rdf2vec -p test_regime 2 -p input_size 10 -p output_file MultiClassificationExperiments.csv; done

# #
# # 100
for i in {1..10}; do papermill MultiClassificationExperiments.ipynb "./experiments_output/MultiClassificationExperiments_rdf2vec_r1_s100_i$i.ipynb" -p emb_method rdf2vec -p test_regime 1 -p input_size 100 -p output_file MultiClassificationExperiments.csv; done
for i in {1..10}; do papermill MultiClassificationExperiments.ipynb "./experiments_output/MultiClassificationExperiments_rdf2vec_r2_s100_i$i.ipynb" -p emb_method rdf2vec -p test_regime 2 -p input_size 100 -p output_file MultiClassificationExperiments.csv; done
#
# 200
for i in {1..10}; do papermill MultiClassificationExperiments.ipynb "./experiments_output/MultiClassificationExperiments_rdf2vec_r1_s200_i$i.ipynb" -p emb_method rdf2vec -p test_regime 1 -p input_size 200 -p output_file MultiClassificationExperiments.csv; done
for i in {1..10}; do papermill MultiClassificationExperiments.ipynb "./experiments_output/MultiClassificationExperiments_rdf2vec_r2_s200_i$i.ipynb" -p emb_method rdf2vec -p test_regime 2 -p input_size 200 -p output_file MultiClassificationExperiments.csv; done

#
#
#

# 
# 10
for i in {1..10}; do papermill MultiClassificationExperiments.ipynb "./experiments_output/MultiClassificationExperiments_bertcode_r1_s10_i$i.ipynb" -p emb_method bertcode -p test_regime 1 -p input_size 10 -p output_file MultiClassificationExperiments.csv; done
for i in {1..10}; do papermill MultiClassificationExperiments.ipynb "./experiments_output/MultiClassificationExperiments_bertcode_r2_s10_i$i.ipynb" -p emb_method bertcode -p test_regime 2 -p input_size 10 -p output_file MultiClassificationExperiments.csv; done
#
# 100
for i in {1..10}; do papermill MultiClassificationExperiments.ipynb "./experiments_output/MultiClassificationExperiments_bertcode_r1_s100_i$i.ipynb" -p emb_method bertcode -p test_regime 1 -p input_size 100 -p output_file MultiClassificationExperiments.csv; done
for i in {1..10}; do papermill MultiClassificationExperiments.ipynb "./experiments_output/MultiClassificationExperiments_bertcode_r2_s100_i$i.ipynb" -p emb_method bertcode -p test_regime 2 -p input_size 100 -p output_file MultiClassificationExperiments.csv; done
#
# 200
for i in {1..10}; do papermill MultiClassificationExperiments.ipynb "./experiments_output/MultiClassificationExperiments_bertcode_r1_s200_i$i.ipynb" -p emb_method bertcode -p test_regime 1 -p input_size 200 -p output_file MultiClassificationExperiments.csv; done
for i in {1..10}; do papermill MultiClassificationExperiments.ipynb "./experiments_output/MultiClassificationExperiments_bertcode_r2_s200_i$i.ipynb" -p emb_method bertcode -p test_regime 2 -p input_size 200 -p output_file MultiClassificationExperiments.csv; done

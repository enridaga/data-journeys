#!/bin/bash

# # VP 10
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p input_size 10 -p dataset BC_VisualisationPreparation.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p input_size 10 -p dataset BC_VisualisationPreparation.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p input_size 10 -p dataset BC_VisualisationPreparation.csv; done
#
# # VP 100
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p input_size 100 -p dataset BC_VisualisationPreparation.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p input_size 100 -p dataset BC_VisualisationPreparation.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p input_size 100 -p dataset BC_VisualisationPreparation.csv; done
#
# # OT 10
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p input_size 10 -p dataset BC_OutputTemporary.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p input_size 10 -p dataset BC_OutputTemporary.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p input_size 10 -p dataset BC_OutputTemporary.csv; done
#
# OT 100
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p input_size 100 -p dataset BC_OutputTemporary.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p input_size 100 -p dataset BC_OutputTemporary.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p input_size 100 -p dataset BC_OutputTemporary.csv; done

# VP 200
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p input_size 200 -p dataset BC_VisualisationPreparation.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p input_size 200 -p dataset BC_VisualisationPreparation.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p input_size 200 -p dataset BC_VisualisationPreparation.csv; done

# OT 200
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p input_size 200 -p dataset BC_OutputTemporary.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p input_size 200 -p dataset BC_OutputTemporary.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p input_size 200 -p dataset BC_OutputTemporary.csv; done


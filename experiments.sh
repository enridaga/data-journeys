#!/bin/bash
source datajourney/bin/activate

# VP 10
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p emb_method rdf2vec -p input_size 10 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p emb_method rdf2vec -p input_size 10 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p emb_method rdf2vec -p input_size 10 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done
#
# VP 100
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p emb_method rdf2vec -p input_size 100 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p emb_method rdf2vec -p input_size 100 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p emb_method rdf2vec -p input_size 100 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done
#
# OT 10
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p emb_method rdf2vec -p input_size 10 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p emb_method rdf2vec -p input_size 10 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p emb_method rdf2vec -p input_size 10 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done
#
# OT 100
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p emb_method rdf2vec -p input_size 100 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p emb_method rdf2vec -p input_size 100 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p emb_method rdf2vec -p input_size 100 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done

# VP 200
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p emb_method rdf2vec -p input_size 200 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p emb_method rdf2vec -p input_size 200 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p emb_method rdf2vec -p input_size 200 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done

# OT 200
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p emb_method rdf2vec -p input_size 200 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p emb_method rdf2vec -p input_size 200 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p emb_method rdf2vec -p input_size 200 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done


################################
# VP 10
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p emb_method bertcode -p input_size 10 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p emb_method bertcode -p input_size 10 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p emb_method bertcode -p input_size 10 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done
#
# VP 100
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p emb_method bertcode -p input_size 100 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p emb_method bertcode -p input_size 100 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p emb_method bertcode -p input_size 100 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done
#
# OT 10
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p emb_method bertcode -p input_size 10 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p emb_method bertcode -p input_size 10 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p emb_method bertcode -p input_size 10 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done
#
# OT 100
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p emb_method bertcode -p input_size 100 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p emb_method bertcode -p input_size 100 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p emb_method bertcode -p input_size 100 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done

# VP 200
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p emb_method bertcode -p input_size 200 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p emb_method bertcode -p input_size 200 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p emb_method bertcode -p input_size 200 -p dataset BC_VisualisationPreparation.csv -p output_file ClassificationExperiments.csv; done

# OT 200
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 1 -p emb_method bertcode -p input_size 200 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 2 -p emb_method bertcode -p input_size 200 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done
for i in {1..10}; do papermill ClassificationExperiments.ipynb /dev/null -p test_regime 3 -p emb_method bertcode -p input_size 200 -p dataset BC_OutputTemporary.csv -p output_file ClassificationExperiments.csv; done

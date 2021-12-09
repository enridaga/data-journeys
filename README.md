# Software and data accompanying the article "Data journeys: knowledge representation and extraction"
Source code of approach, algorithms, and experimental results.

## Python environment
All experiments are executed on a MacOS-X 10.15.7 with Python 3.8, configured via Conda in folder `dj-py3.8`.
```
$ conda create --name dj-py3.9 python=3.9
```
To activate this environment, use
```
$ source activate dj-py3.9
```
To deactivate an active environment, use
```
$ conda deactivate
```
Initialise the bash shell with:
```
$ conda init bash
```

Prerequirements:
```
$ conda activate dj-py3.9
$ conda install networkx
$ pip install rdflib graphviz pygraphviz jupyterlab
$ pip install pandas numpy sklearn transformers gensim aiohttp
$ pip install pyrdf2vec
$ pip install papermill

```
Install Jupyter
## DATA JOURNEY ONTOLOGY (DJO)
The ontology is in folder `ontology/datajourneys.ttl`

## Approach for extracting data journeys / Evaluation

Kernels downloaded from Kaggle: [kernels.zip](kernels.zip). The content of the file is expanded in folder `kernels/`.

### Datanode graph extraction

The algorithm presented in Listing 1 is implemented in file `datajourney.py`. 

The process is divided in two steps:

1) `(dj-py3.8) $ python process_kernels.py` generates a datanode representation in DOT format. Output is saved in folders `sources/` (python code only from notebboks) and `graphs/` (directed graph in DOT format). The first includes the source code extracted from the notebook and the `graphs/` folder includes the generated datanode graphs in DOT format.
2) `(dj-py3.8) $ python generate_rdf.py` reengineers the content to RDF. Output is saved in folder `rdf/` and `graphs/`.

The `rdf/` folder includes the datanode graphs extracted.

### Knowledge expansion

The Frequent Activity Table (FAT) is produced running the following SPARQL query on a Triple Store containing all the RDF files generated in the previous step.
The triple store used is Blazegraph, in folder `blazegraph/`.

1) Load the RDF files with script `cd blazegraph && ./bulk_load.sh`
2) Start blazegraph: `$ java -jar -Xmx4G blazegraph.jar`, UI can be accessed from the browser (follow the instructions in terminal)
3) The Frequent Activity Table (FAT) is generated with a SPARQL query counting the number of occurrences of the properties in the graph:
```
SELECT ?arc (COUNT(*) as ?count)
WHERE {
  [] ?arc []
} 
group by ?arc
order by desc(?count)
```
4) The Frequent Activity Table (FAT) annotated with activity types was produced with a Google Spreadsheet, accessible at this URL: https://docs.google.com/spreadsheets/d/1zx_XK9VhEtgxFFXpFy9RYzX5MDxZxZZDnqxOqvkXoDQ/edit?usp=sharing

5) Rules are generated with notebook [Process ARCS rules.ipynb](<Process ARCS rules.ipynb>). SPARQL Construct queries are reported in file `activity_rules.json`.
6) The training dataset is then produced by querying the triple store for instances of the arcs reported in Table 2, using the following SPARQL query:

```
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
PREFIX owl:  <http://www.w3.org/2002/07/owl#> 
PREFIX dj:   <http://purl.org/dj/> 
PREFIX : <http://purl.org/datajourneys/> 

SELECT DISTINCT ?Notebook ?Node ?Arc ?Label ?Team
WHERE {
  BIND ( STRBEFORE(SUBSTR(STR(?Node), 27), "#") AS ?Notebook ) .
  { BIND(dj:print as ?Arc) . BIND(":Visualisation" AS ?Label) . ?Node ?Arc [] . }
  UNION
  { BIND(dj:append as ?Arc) . BIND(":Preparation" AS ?Label) . ?Node ?Arc [] . }
  UNION
  { BIND(dj:plot as ?Arc) . BIND("T1" AS ?Team) . BIND(":Visualisation" AS ?Label) . ?Node ?Arc [] . }
  UNION
  { BIND(dj:Add as ?Arc) . BIND("T1" AS ?Team) . BIND(":Preparation" AS ?Label) . ?Node ?Arc [] . }
  UNION
  { BIND(dj:importedBy as ?Arc) . BIND("T1" AS ?Team) . BIND(":Reuse" AS ?Label) . ?Node ?Arc [] . }
  UNION
  { BIND(dj:read_csv as ?Arc) . BIND("T1" AS ?Team) . BIND(":Movement" AS ?Label) . ?Node ?Arc [] . }
  UNION
  { BIND(dj:to_csv as ?Arc) . BIND("T1" AS ?Team) . BIND(":Movement" AS ?Label) . ?Node ?Arc [] . }
  UNION
  { BIND(dj:predict as ?Arc) . BIND("T1" AS ?Team) . BIND(":Analysis" AS ?Label) . ?Node ?Arc [] . }
  UNION
  { BIND(dj:tanh as ?Arc) . BIND("T1" AS ?Team) . BIND(":Analysis" AS ?Label) . ?Node ?Arc [] . }
  UNION
  { BIND(dj:subplots as ?Arc) . BIND("T2" AS ?Team) . BIND(":Visualisation" AS ?Label) . ?Node ?Arc [] . }
  UNION
  { BIND(dj:iteratorOf as ?Arc) . BIND("T2" AS ?Team) . BIND(":Preparation" AS ?Label) . ?Node ?Arc [] . }
  UNION
  { BIND(dj:_argToVar as ?Arc) . BIND("T2" AS ?Team) . BIND(":Reuse" AS ?Label) . ?Node ?Arc [] . }
  UNION
  { BIND(dj:copy as ?Arc) . BIND("T1" AS ?Team) . BIND(":Movement" AS ?Label) . ?Node ?Arc [] . }
  UNION
  { BIND(dj:fit as ?Arc) . BIND("T1" AS ?Team) . BIND(":Analysis" AS ?Label) . ?Node ?Arc [] . }
}
```
The output is saved in file `MultiClassification.csv` (please note that the Team column was not used in the final experiments, as we opted for random sampling).

#### Machine Learning Application
The model used in the machine learning application step was produced by configuring the [MultiClassificationExperiments.ipynb](MultiClassificationExperiments.ipynb) notebook with the best performing parameters (see next section). The used model is available: `./models/MLPClassifier_2_1000_rdf2vec.clf`.
To re-build the models, execute the script `build_classifier.sh`.

### Experiments with machine learning algorithms

Experiments are prepared with Python 3.8 installed via PyEnv in folder `dj-py3.8` and executed with Papermill through a bash script on a MacOS-X. 
The experiment is designed in Notebook [MultiClassificationExperiments.ipynb](MultiClassificationExperiments.ipynb).
The training file is `MultiClassification.csv`.

Parameters are:
1) `emb_method` rdf2vec | bertcode
2) `test_regime` 1 | 2 
3) `input_size` number of notebooks

The script for reproduction is `multi-experiments.sh`. We report one line of the script, for explanatory purposes:
```
for i in {1..10}; do papermill MultiClassificationExperiments.ipynb "./experiments_output/MultiClassificationExperiments_rdf2vec_r1_s10_i$i.ipynb" -p emb_method rdf2vec -p test_regime 1 -p input_size 10 -p output_file MultiClassificationExperiments.csv; done
```
The script repeats the experiments with different parameters: 10 to 200 randomly choosen notebooks, embedding method `rdf2vec` or `bertcode` and test regime `1` or `2`.

Results are saved to file [MultiClassificationExperiments.csv](MultiClassificationExperiments).

Results can be explored and analysed in [AnalyseResultsMulti.ipynb](AnalyseResultsMulti.ipynb).
### Knowledge compression
This phase is performed in notebook [DataJourneyGenerator.ipynb](DataJourneyGenerator.ipynb). This notebook was executed on each of the input notebook. Output is in folder `datajourneys/`. 

### Guide example
The resulting notebooks are in folder `datajourneys/`. The guide example discussed in the paper is reported in the following files:

- random-forests: digraph generated in the first step of Datanode graph extraction
- random-forests.png: the datanode graph without activity annotations (before running the FAR)
- random-forests_DN.digraph: the datanode graph with activity annotations (after running the FAR) -- Digraph
- random-forests_DN.png: the datanode graph with activity annotations (after running the FAR) -- PNG image
- random-forests_DN.svg: the datanode graph with activity annotations (after running the FAR) -- SVG image
- random-forests_DJ.digraph: the digraph representation of the output Activity Graph
- random-forests_DJ.png: a graph representation of the output Activity Graph -- PNG image
- random-forests_DJ.svg: a graph representation of the output Activity Graph -- SVG image
- random-forests.ttl: the complete Data Journey (datanode graph + activity graph)

The same files are available for each one of the Notebooks used in our experiments, in the `datajourneys/` folder.

### Compression rate

Statistics are collected from the data files in `datajourneys/` with the script `compression_rate.sh`. The data reported in the paper is in `compression_rate.csv`.

The diagram in the paper was produced with a Google Spreadsheet, accessible at this link: https://docs.google.com/spreadsheets/d/1zx_XK9VhEtgxFFXpFy9RYzX5MDxZxZZDnqxOqvkXoDQ/edit?usp=sharing

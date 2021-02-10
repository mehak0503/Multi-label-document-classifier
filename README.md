# INTRODUCTION

An ensemble classifier on data containing pdf, docx and doc files in folder structures with the following components :-
1. Classification based on document text using NLP
2. Classification based on structure of document hierarchy
3. classification based on document properties as file name, type, size


# TECHNOLOGIES USED

- Python
- Spark NLP
- Neo4j
- Machine Learning
- Data Science


# DEPENDENCIES

- Python 3.7
- pip install -r requirements.txt



# STEPS tO RUN

1. Run predict.py in the folder containing models

2. Input complete path of file from root. The classifier takes input as list so that any number of files can be given as input in one go. 

3. It will output the filenames along with associated labels.


# FILES

- trainClassifiers.py :- Trains the classifiers and uses utils.py, allComponents.py
- prepareDataset.py :- Extracts features from the input file i.e. text, type, size, adjacent files.
- utils.py :- Creates pySpark pipelines and train-test data
- allComponents :- Provides individual components for pySpark pipeline
- predict.py :- Takes filename input and extracts features using prepareDataset.py followed by generating predicitions and labels. 


# DATA FILES

 - dataset.csv :- Contains text, type, size, path information for all the files in Record-Management-Data/Data. This is used as training data set for text based  and feature based classifier.
 
 - neo-data.csv :- Contains nodes, properties, adjacent nodes data for Neo4j database. This is used as training dataset for neo4j classifier.
 
 - PathCategorisation.csv :- Contains split path components for entire path and used along with training set for feature based classifier.
 
 - DocumentCategorisation.csv :- Contains label components for all files and used along with training set for supervised learning.



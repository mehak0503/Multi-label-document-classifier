import sparknlp
spark = sparknlp.start() # for GPU training >> sparknlp.start(gpu = True) # for Spark 2.3 =>> sparknlp.start(spark23 = True)
from pyspark.sql import SQLContext
from pyspark import SparkContext
from utils import *
sc =SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

def create_dataset(dataFile, labelFile):
    """
        Reads csv files with dataset and labels to generate pySpark dataframe
        Input: dataFile, labelFile
        Output: pySpark dataframe
    """    
    data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(dataFile)
    data1 = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(labelFile)
    df = data.join(data1, (data['filename'] == data1['Document Name']))
    oldColumns = ['filename','filepath','filetype','filesize','filetext','translatedtext','Document Name','Location','Category 1 (Mandatory)','Category 2 (Optional)','Category 3 (Optional)']
    newColumns = ['filename','filepath','filetype','filesize','filetext','translatedtext','DocumentName','Location','Category1(Mandatory)','Category2(Optional)','Category3(Optional)']
    df = reduce(lambda data, idx: data.withColumnRenamed(oldColumns[idx], newColumns[idx]), range(len(oldColumns)), df)
    drop_list = ['filename', 'filepath','filetext','Category3(Optional)']
    result = df.select([column for column in df.columns if column not in drop_list])
    result = result.na.fill("NotSpecified")#Fill empty
    #result.show(5)
    return result

def create_neo4j_dataset(dataFile, labelFile):
    """
        Reads csv files with dataset for neo4j and labels to generate pySpark dataframe
        Input: dataFile, labelFile
        Output: pySpark dataframe
    """
    data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(dataFile)
    data1 = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(labelFile)
    df = data.join(data1, (data['nodes'] == data1['Document Name']))
    oldColumns = ['Document Name','Category 1 (Mandatory)','Category 2 (Optional)','Category 3 (Optional)']
    newColumns = ['DocumentName','Category1(Mandatory)','Category2(Optional)','Category3(Optional)']
    df = reduce(lambda data, idx: data.withColumnRenamed(oldColumns[idx], newColumns[idx]), range(len(oldColumns)), df)
    drop_list = ['Category3(Optional)']
    result = df.select([column for column in df.columns if column not in drop_list])
    result = result.na.fill("NotSpecified")#Fill empty
    #result.show(5)
    return result


def train_text_classifier(choice,dataframe,outputCol,fileName):
    """
        Trains text based classifier and saves model along with generation of classification report
        Input: choice,dataframe,outputCol,fileName
        Output: classification report, model
    """
    tc = trainClassifier()
    inputCol = "translatedtext"
    pipeline = tc.get_text_pipeline(choice,inputCol,outputCol)
    processed_df = pipeline.fit(df).transform(df)
    trainingData,testData = tc.train_test_split(processed_df)
    trainingData,testData = tc.process_train_test_data(trainingData,testData,"features")
    tc.get_classification_report(tc.logistic_regression(trainingData,testData,fileName),"features","label")

def train_features_classifier(dataframe,outputCol,fileName):
    """
        Trains features based classifier and saves model along with generation of classification report
        Input: dataframe,outputCol,fileName
        Output: classification report, model
    """
    tc = trainClassifier()
    inputCol = "path"
    pipeline = tc.get_features_pipeline(outputCol,inputCol,"filetype","filesize")
    processed_df = pipeline.fit(df).transform(df)
    trainingData,testData = tc.train_test_split(processed_df)
    tc.get_classification_report(tc.logistic_regression(trainingData,testData,fileName),"features","label")

def train_neo4j_classifier(dataframe,outputCol,fileName):
    """
        Trains neo4j based classifier and saves model along with generation of classification report
        Input: dataframe,outputCol,fileName
        Output: classification report, model
    """
    tc = trainClassifier()
    inputCol = "nodes"
    pipeline = tc.get_neo4j_pipeline(outputCol,inputCol,"adj_nodes")
    processed_df = pipeline.fit(df).transform(df)
    trainingData,testData = tc.train_test_split(processed_df)
    tc.get_classification_report(tc.logistic_regression(trainingData,testData,fileName),"features","label")

if __name__ == "__main__":
    df = create_dataset('dataset.csv','Document Categorisation.csv')
    train_text_classifier(3,df,"Category1(Mandatory)","textClassifierLabel1.model")
    train_text_classifier(3,df,"Category2(Optional)","textClassifierLabel2.model")
    df = create_dataset('dataset.csv','PathCategorisation.csv')
    train_features_classifier(df,"Category1(Mandatory)","featuresClassifierLabel1.model")
    train_features_classifier(df,"Category2(Optional)","featuresClassifierLabel2.model")
    df = create_neo4j_dataset('neo_data.csv','Document Categorisation.csv')
    train_neo4j_classifier(df,"Category1(Mandatory)","neo4jClassifierLabel1.model")
    train_neo4j_classifier(df,"Category2(Optional)","neo4jClassifierLabel2.model")    

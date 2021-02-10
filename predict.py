from pyspark.ml.classification import OneVsRestModel,LogisticRegressionModel
import sparknlp
spark = sparknlp.start() # for GPU training >> sparknlp.start(gpu = True) # for Spark 2.3 =>> sparknlp.start(spark23 = True)
from pyspark.sql import SQLContext
from pyspark import SparkContext
from utils import *
sc =SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
from pyspark.sql.functions import *
from collections import defaultdict 
from prepareDataset import *

def return_label1(value):
    """
        Returns label 1 corresponding to predicted class
        Input: predicted class
        Output: label 
    """
    labels =  {24.0: 'Report',
    19.0: 'Policy',
    2.0: 'Aged Care Quality Standards',
    21.0: 'Human Resources',
    4.0: 'Dept Health',
    25.0: 'Advanced Care Planning',
    26.0: 'NOUS Report',
    22.0: 'Consumer Experience',
    0.0: 'Audit',
    5.0: 'COVID-19',
    15.0: 'Newsletter',
    6.0: 'Handbook',
    13.0: 'Other',
    10.0: 'Notice of Collection',
    16.0: 'Key Changes for Providers',
    12.0: 'Accreditation',
    18.0: 'Memorandum of Understanding',
    1.0: 'Draft Guidance Material',
    20.0: 'Sector Performance',
    17.0: 'Statements ',
    14.0: 'Corporate Plan',
    11.0: 'Self Assessment',
    9.0: 'Regulatory Bulletin',
    23.0: 'Clinical Care Standard',
    7.0: 'Standards Guidance Reference Group',
    8.0: 'Annual Report',
    3.0: 'Charter of Aged Care Rights'}
    return labels[value]

def return_label2(value):
    """
        Returns label 2 corresponding to predicted class
        Input: predicted class
        Output: label 
    """
    labels =  {47.0: 'Maltese',
    61.0: 'St Vincent�s Care Services Heathcote',
    15.0: 'A.G.Eastwood Hostel',
    18.0: 'Application Document',
    8.0: 'Fact sheet',
    2.0: 'Storyboard',
    49.0: 'Croatian',
    1.0: 'Feedback',
    33.0: 'Hungarian',
    20.0: '70 Lowe Street',
    41.0: 'Perry Park Hostel',
    34.0: 'Perry Park Nursing Home',
    22.0: 'Quality and Safety',
    7.0: 'Acacia Living Group Meadow Springs Aged',
    56.0: 'COVID-19',
    13.0: 'Mingarra Hostel',
    23.0: 'A H Orr Lodge',
    66.0: 'Portuguese',
    36.0: 'Highercombe',
    31.0: 'RSL War Veterans Home Mandurah',
    25.0: 'RSL Menora Gardens Aged Care Facility',
    43.0: 'Colton Court Nursing',
    54.0: 'Simplified Chinese',
    0.0: 'NotSpecified',
    28.0: 'Polish',
    29.0: 'Kapara Nursing Home',
    9.0: 'ACDMA Aged Hostel',
    4.0: 'Booklet',
    50.0: 'Executive Remuneration',
    64.0: 'St Elizabeth Home',
    26.0: 'Greek',
    3.0: 'Poster',
    16.0: 'Abbeyfield House Hostel',
    62.0: 'Latvian',
    40.0: 'German',
    60.0: 'Minda Nursing Home',
    10.0: 'Report',
    52.0: 'Traditional Chinese',
    6.0: 'A Little Yarn',
    11.0: 'ACH Group Residential Care',
    55.0: 'Assault',
    17.0: '501 Care Services',
    39.0: 'The Abbey Nursing Home',
    45.0: 'Serbian',
    51.0: 'English',
    46.0: 'Mental Health',
    30.0: 'Macedonian',
    53.0: 'Spanish',
    72.0: 'Sexual Assessment',
    32.0: 'Consumer Experience',
    77.0: 'French',
    37.0: 'Dutch',
    63.0: 'Ukrainian',
    12.0: 'Abernethy Nursing Home',
    68.0: 'Stakeholder Report',
    70.0: 'Restrain Scenarios',
    24.0: 'Complaints',
    19.0: 'Abberfield Aged Care Facility',
    5.0: 'Communique',
    38.0: 'Milpara Aged Care Facility',
    57.0: 'Vietnamese',
    35.0: 'Arabic',
    48.0: 'Italian',
    67.0: 'Medical',
    58.0: 'Dementia',
    73.0: 'Diabetes',
    27.0: 'Russian',
    69.0: 'Poster ',
    74.0: 'Spiritual Care',
    71.0: 'Quality Care',
    14.0: 'Abel Tasman Village',
    65.0: 'HammondCare � Miranda',
    44.0: 'Korean',
    59.0: 'Food and Nutrition',
    21.0: 'Abbey House Aged Care',
    42.0: 'Hindi',
    76.0: 'Turkish',
    75.0: 'Newmarch House'}
    return labels[value]

def predict_label1(data):
    """
        Predicts label1 using weighted average of prediction from 
        3 classifiers :- Text based, feature based, neo4j data based
        Input: dataframe
        Output: pySpark dataframe with prediction and probabilities from 3 classifiers 
    """
    tc = trainClassifier()
    #Prediction from text based classifier
    inputCol = "translatedtext"
    model = LogisticRegressionModel.load('textClassifierLabel1.model')
    pipeline = tc.get_text_predict_pipeline(3,inputCol)
    processed_df = pipeline.fit(data).transform(data)
    res = model.transform(processed_df).select('filename','probability','prediction')
    res = res.withColumnRenamed('probability','probability_text').withColumnRenamed('prediction','prediction_text')
    #Prediction from features based classifier
    model = LogisticRegressionModel.load('featuresClassifierLabel1.model')
    inputCol = "filepath"
    pipeline = tc.get_features_predict_pipeline(inputCol,"filetype","filesize")
    processed_df = pipeline.fit(data).transform(data)
    res1 = model.transform(processed_df).select('filename','probability','prediction')
    res1 = res1.withColumnRenamed('probability','probability_features').withColumnRenamed('prediction','prediction_features').\
            withColumnRenamed('filename','filename1')
    res = res.join(res1, res1.filename1 == res.filename)
    #Prediction from neo4j classifier
    model = LogisticRegressionModel.load('neo4jClassifierLabel1.model')
    inputCol = "filename"
    pipeline = tc.get_neo4j_predict_pipeline(inputCol,"adj_nodes")
    processed_df = pipeline.fit(data).transform(data)
    res2 = model.transform(processed_df).select('filename','probability','prediction')
    res2 = res2.withColumnRenamed('probability','probability_neo4j').withColumnRenamed('prediction','prediction_neo4j').\
            withColumnRenamed('filename','filename2')
    res = res.join(res2, res2.filename2 == res.filename)
    print(res.show())
    values = res.collect()
    return values

def predict_label2(data):
    """
        Predicts label2 using weighted average of prediction from 
        3 classifiers :- Text based, feature based, neo4j data based
        Input: dataframe
        Output: pySpark dataframe with prediction and probabilities from 3 classifiers 
    """
    tc = trainClassifier()
    #Prediction from text based classifier
    inputCol = "translatedtext"
    model = LogisticRegressionModel.load('textClassifierLabel2.model')
    pipeline = tc.get_text_predict_pipeline(3,inputCol)
    processed_df = pipeline.fit(data).transform(data)
    res = model.transform(processed_df).select('filename','probability','prediction')
    res = res.withColumnRenamed('probability','probability_text').withColumnRenamed('prediction','prediction_text')
    #Prediction from features based classifier
    model = LogisticRegressionModel.load('featuresClassifierLabel2.model')
    inputCol = "filepath"
    pipeline = tc.get_features_predict_pipeline(inputCol,"filetype","filesize")
    processed_df = pipeline.fit(data).transform(data)
    res1 = model.transform(processed_df).select('filename','probability','prediction')
    res1 = res1.withColumnRenamed('probability','probability_features').withColumnRenamed('prediction','prediction_features').\
            withColumnRenamed('filename','filename1')
    res = res.join(res1, res1.filename1 == res.filename)
    #Prediction from neo4j based classifier
    model = LogisticRegressionModel.load('neo4jClassifierLabel2.model')
    inputCol = "filename"
    pipeline = tc.get_neo4j_predict_pipeline(inputCol,"adj_nodes")
    processed_df = pipeline.fit(data).transform(data)
    res2 = model.transform(processed_df).select('filename','probability','prediction')
    res2 = res2.withColumnRenamed('probability','probability_neo4j').withColumnRenamed('prediction','prediction_neo4j').\
            withColumnRenamed('filename','filename2')
    res = res.join(res2, res2.filename2 == res.filename)
    print(res.show())
    values = res.collect()
    return values

def prediction(data):
    """
        Predicts labels using weighted average of prediction from 
        3 classifiers :- Text based, feature based, neo4j data based
        Performs weighted average score for 3 classifiers in 1:2:2 ratio
        Input: dataframe
        Output: labels corresponding to files
    """
    values1 = predict_label1(data)
    values2 = predict_label2(data)
    result = defaultdict(list)
    for i in values1:
        result[i[0]].append(return_label1(((i[1]+(i[4]*2)+(i[8]*2))/5).argmax()))
    for i in values2:
        result[i[0]].append(return_label2(((i[1]+(i[4]*2)+(i[8]*2))/5).argmax()))
    for i in result:
        print(i,result[i])

if __name__=="__main__":
    dt = prepareData()
    #files = dt.read_files('Record-Management-Data/Data')
    inp = input('Enter filename (with complete path): ')
    files = [inp]
    df = dt.form_dataset(files,'testData.csv')
    data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('testData.csv')
    print(data.show())
    prediction(data)
    
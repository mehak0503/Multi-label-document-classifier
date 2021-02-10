import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
import pandas as pd
from allComponents import Components
from functools import reduce
from pyspark.sql.functions import udf
from pyspark.ml.classification import OneVsRest,LogisticRegression,NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

class trainClassifier:
    def train_test_split(self,dataframe):
        """
            Split dataframe into 80:20 train-test data
            Input: dataframe
            Output: trainingData,testData
        """
        (trainingData, testData) = dataframe.randomSplit([0.8, 0.2], seed = 100)
        return trainingData, testData
        
    def process_train_test_data(self,trainingData,testData,outputCol):
        """
            Check and eliminate zero vectors in train-test data
            Input: trainingData,testData, outputCol 
            Output: trainingData,testData
        """
        @udf("long")
        def num_nonzeros(v):
            return v.numNonzeros()
        testData = testData.where(num_nonzeros(outputCol) != 0)
        trainingData = trainingData.where(num_nonzeros(outputCol) != 0)
        return trainingData,testData

    def get_classification_report(self,dataframe,inputCol,outputCol):
        """
            Generates sklearn classification report
            Input: dataframe,inputCol,outputCol 
            Output: accuracy, classification report
        """
        evaluator = MulticlassClassificationEvaluator(predictionCol=outputCol)
        print("MulticlassEvaluator score: ",evaluator.evaluate(dataframe))
        df = dataframe.select(inputCol,outputCol,"prediction").toPandas()
        print(classification_report(df.label, df.prediction))
        print(accuracy_score(df.label, df.prediction))

    def logistic_regression(self,trainingData,testData,fileName):
        """
            Trains, save and transform logistic regression model
            Input: trainingData,testData,fileName to save model 
            Output: predictions result on testData and model
        """
        lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0)
        model = lr.fit(trainingData)
        model.save(fileName)
        return model.transform(testData)

    def oneRest(self,trainingData,testData,fileName):
        """
            Trains, save and transform onevsrest model
            Input: trainingData,testData,fileName to save model 
            Output: predictions result on testData and model
        """
        lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)
        ovr = OneVsRest(classifier=lr)
        model = ovr.fit(trainingData)
        model.save(fileName)
        return model.transform(testData)

    #Features Classifier
    def get_features_pipeline(self,outCol,*inputCol):
        """
            Returns pipeline to train feature classifier
            Input: ouputCol,inputCol(tuple)
            Output: pySpark pipeline
        """
        c = Components()
        allStages = [c.getDocumentAssembler(inputCol[0],"document"),c.getTokenizer("document","tokens"),\
                    c.getNormalizer("tokens","normalized"),\
                    c.getFinisher("normalized","finished"),\
                    c.getTf("finished","tf"),c.getIdf("tf","locFeature"),\
                    c.getStringIndexer(inputCol[1],"typeFeature"),\
                    c.getVectorAssembler(["locFeature","typeFeature",inputCol[2]],"features"),\
                    c.getStringIndexer(outCol,"label")]
        return Pipeline(stages=allStages)
        
    #Neo4j based Classifier    
    def get_neo4j_pipeline(self,outCol,*inputCol):
        """
            Returns pipeline to train neo4j classifier
            Input: ouputCol,inputCol(tuple)
            Output: pySpark pipeline
        """
        c = Components()
        allStages = [c.getDocumentAssembler(inputCol[0],"document"),c.getTokenizer("document","tokens"),\
                    c.getNormalizer("tokens","normalized"),\
                    c.getFinisher("normalized","finished"),\
                    c.getDocumentAssembler(inputCol[1],"document1"),\
                    c.getTokenizer("document1","tokens1"),\
                    c.getNormalizer("tokens1","normalized1"),\
                    c.getFinisher("normalized1","finished1"),\
                    c.getTf("finished","tf"),c.getIdf("tf","locFeature"),\
                    c.getTf("finished1","tf1"),c.getIdf("tf1","adjFeature"),\
                    c.getVectorAssembler(["locFeature","adjFeature"],"features"),\
                    c.getStringIndexer(outCol,"label")]
        return Pipeline(stages=allStages)
    
    #Text classifier
    def get_text_pipeline(self,choice,inputCol,outCol):
        """
            Returns pipeline to train text based classifier
            Input: ouputCol,inputCol(tuple)
            Output: pySpark pipeline
        """
        c = Components()
        allStages = [c.getDocumentAssembler(inputCol,"document"),c.getTokenizer("document","tokens"), 
                    c.getNormalizer("tokens","normalized"),c.getStopWordCleaner("normalized","cleaned"), 
                    c.getStemmer("cleaned","stemmed")]
        if choice==0:#Glove Embeddings
            allStages.extend([c.getGloveEmbeddings(["document","stemmed"],"embeddings"),\
                            c.getEmbeddingSentence(["document", "embeddings"],"sentence_embeddings"),\
                            c.getEmbeddingFinisher("sentence_embeddings","finished_sentence_embeddings"),\
                            c.getExplodeVectors("finished_sentence_embeddings","features"),\
                            c.getStringIndexer(outCol,"label")])
            return Pipeline(stages=allStages)
        elif choice==1:#BERT Embeddings
            allStages.extend([c.getBERTEmbeddings(["document","stemmed"],"embeddings"),\
                            c.getEmbeddingSentence(["document", "embeddings"],"sentence_embeddings"),\
                            c.getEmbeddingFinisher("sentence_embeddings","finished_sentence_embeddings"),\
                            c.getExplodeVectors("finished_sentence_embeddings","features"),\
                            c.getStringIndexer(outCol,"label")])
            return Pipeline(stages=allStages)    
        elif choice==2:#ELMO Embeddings
            allStages.extend([c.getELMOEmbeddings(["document","stemmed"],"embeddings"),\
                            c.getEmbeddingSentence(["document", "embeddings"],"sentence_embeddings"),\
                            c.getEmbeddingFinisher("sentence_embeddings","finished_sentence_embeddings"),\
                            c.getExplodeVectors("finished_sentence_embeddings","features"),\
                            c.getStringIndexer(outCol,"label")])
            return Pipeline(stages=allStages)
        elif choice==3:#USE Embeddings
            allStages = [c.getDocumentAssembler(inputCol,"document"),\
                            c.getUSEEmbeddings("document","embeddings"),\
                            c.getEmbeddingFinisher("embeddings","finished_sentence_embeddings"),\
                            c.getExplodeVectors("finished_sentence_embeddings","features"),\
                            c.getStringIndexer(outCol,"label")]
            return Pipeline(stages=allStages)

    #Features Classifier
    def get_features_predict_pipeline(self,*inputCol):
        """
            Returns pipeline to test and predict using feature classifier
            Input: inputCol(tuple)
            Output: pySpark pipeline
        """
        c = Components()
        allStages = [c.getDocumentAssembler(inputCol[0],"document"),\
                    c.getTokenizer("document","tokens"),\
                    c.getNormalizer("tokens","normalized"),\
                    c.getFinisher("normalized","finished"),\
                    c.getTf("finished","tf"),c.getIdf("tf","locFeature"),\
                    c.getVectorAssembler(["locFeature",inputCol[1],inputCol[2]],"features")]
        return Pipeline(stages=allStages)
        
    #Neo4j based Classifier    
    def get_neo4j_predict_pipeline(self,*inputCol):
        """
            Returns pipeline to test and predict using neo4j classifier
            Input: inputCol(tuple)
            Output: pySpark pipeline
        """
        c = Components()
        allStages = [c.getDocumentAssembler(inputCol[0],"document"),c.getTokenizer("document","tokens"),\
                    c.getNormalizer("tokens","normalized"),\
                    c.getFinisher("normalized","finished"),\
                    c.getDocumentAssembler(inputCol[1],"document1"),\
                    c.getTokenizer("document1","tokens1"),\
                    c.getNormalizer("tokens1","normalized1"),\
                    c.getFinisher("normalized1","finished1"),\
                    c.getTf("finished","tf"),c.getIdf("tf","locFeature"),\
                    c.getTf("finished1","tf1"),c.getIdf("tf1","adjFeature"),\
                    c.getVectorAssembler(["locFeature","adjFeature"],"features")]
        return Pipeline(stages=allStages)
    
    #Text classifier
    def get_text_predict_pipeline(self,choice,inputCol):
        """
            Returns pipeline to test and predict using text based classifier
            Input: inputCol(tuple)
            Output: pySpark pipeline
        """
        c = Components()
        allStages = [c.getDocumentAssembler(inputCol,"document"),c.getTokenizer("document","tokens"), 
                    c.getNormalizer("tokens","normalized"),c.getStopWordCleaner("normalized","cleaned"), 
                    c.getStemmer("cleaned","stemmed")]
        if choice==0:#Glove Embeddings
            allStages.extend([c.getGloveEmbeddings(["document","stemmed"],"embeddings"),\
                            c.getEmbeddingSentence(["document", "embeddings"],"sentence_embeddings"),\
                            c.getEmbeddingFinisher("sentence_embeddings","finished_sentence_embeddings"),\
                            c.getExplodeVectors("finished_sentence_embeddings","features")])
            return Pipeline(stages=allStages)
        elif choice==1:#BERT Embeddings
            allStages.extend([c.getBERTEmbeddings(["document","stemmed"],"embeddings"),\
                            c.getEmbeddingSentence(["document", "embeddings"],"sentence_embeddings"),\
                            c.getEmbeddingFinisher("sentence_embeddings","finished_sentence_embeddings"),\
                            c.getExplodeVectors("finished_sentence_embeddings","features")])
            return Pipeline(stages=allStages)    
        elif choice==2:#ELMO Embeddings
            allStages.extend([c.getELMOEmbeddings(["document","stemmed"],"embeddings"),\
                            c.getEmbeddingSentence(["document", "embeddings"],"sentence_embeddings"),\
                            c.getEmbeddingFinisher("sentence_embeddings","finished_sentence_embeddings"),\
                            c.getExplodeVectors("finished_sentence_embeddings","features")])
            return Pipeline(stages=allStages)
        elif choice==3:#USE Embeddings
            allStages = [c.getDocumentAssembler(inputCol,"document"),\
                            c.getUSEEmbeddings("document","embeddings"),\
                            c.getEmbeddingFinisher("embeddings","finished_sentence_embeddings"),\
                            c.getExplodeVectors("finished_sentence_embeddings","features")]
            return Pipeline(stages=allStages)

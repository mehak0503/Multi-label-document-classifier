from pyspark.ml.feature import *
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.sql.functions import explode
from pyspark.ml.linalg import Vectors

class Components:
    def getDocumentAssembler(self,inputCols,outputCols):
        """
        inputCols(str)
        outputCols(str)
        """
        document_assembler = DocumentAssembler().setInputCol(inputCols).setOutputCol(outputCols)
        return document_assembler


    def getTokenizer(self,inputCols,outputCols):
        """
        inputCols(str)
        outputCols(str)
        """
        tokenizer = Tokenizer().setInputCols([inputCols]).setOutputCol(outputCols)
        return tokenizer


    def getNormalizer(self,inputCols,outputCols):
        """
        inputCols(str)
        outputCols(str)
        """
        normalizer = Normalizer().setInputCols([inputCols]).setOutputCol(outputCols).setLowercase(True)
        return normalizer

    def getStopWordCleaner(self,inputCols,outputCols):
        """
        inputCols(str)
        outputCols(str)
        """
        stopwords_cleaner = StopWordsCleaner().setInputCols([inputCols]).setOutputCol(outputCols).setCaseSensitive(False)
        return stopwords_cleaner

    def getStemmer(self,inputCols,outputCols):
        """
        inputCols(str)
        outputCols(str)
        """
        stemmer = Stemmer().setInputCols([inputCols]).setOutputCol(outputCols)
        return stemmer
        
    def getFinisher(self,inputCols,outputCols):
        """
        inputCols(str)
        outputCols(str)
        """
        finisher = Finisher().setInputCols([inputCols]).setOutputCols([outputCols]).setOutputAsArray(True).setCleanAnnotations(False)
        return finisher

    def getSentenceDetector(self,inputCols,outputCols):
        """
        inputCols(str)
        outputCols(str)
        """
        sentenceDetector = SentenceDetector().setInputCols([inputCols]).setOutputCol([outputCols])
        return sentenceDetector

    def getTokenAssembler(self,inputCols,outputCols):
        """
        inputCols(list): sentenceCol, TokensCol
        outputCols(str)
        """
        tokenassembler = TokenAssembler().setInputCols(["sentences", "cleanTokens"]).setOutputCol("clean_text")
        return tokenassembler

    def getLemmatizer(self,inputCols,outputCols):
        """
        inputCols(str)
        outputCols(str)
        """
        lemmatizer = Lemmatizer().setInputCols([inputCols]).setOutputCol(outputCols)
        return lemmatizer

    def getCountVectorizer(self,inputCols,outputCols):
        """
        inputCols(str)
        outputCols(str)
        """
        countVect = CountVectorizer(inputCol=inputCols, outputCol=outputCols, vocabSize=10000, minDF=5)
        return countVect

    def getTf(self,inputCols,outputCols):
        """
        inputCols(str)
        outputCols(str)
        """
        hashingTF = HashingTF(inputCol=inputCols, outputCol=outputCols, numFeatures=10000)
        return hashingTF

    
    def getIdf(self,inputCols,outputCols):
        """
        inputCols(str)
        outputCols(str)
        """
        idf = IDF(inputCol=inputCols, outputCol=outputCols, minDocFreq=5) #minDocFreq: remove sparse terms
        return idf

    def getStringIndexer(self,inputCols,outputCols):
        """
        inputCols(str)
        outputCols(str)
        """
        label_stringIdx = StringIndexer(inputCol = inputCols, outputCol =outputCols)
        return label_stringIdx

    def getGloveEmbeddings(self,inputCols,outputCols):
        """
        inputCols(list): documentCol, tokenCol
        outputCols(str)
        """
        glove_embeddings = WordEmbeddingsModel().pretrained().setInputCols(inputCols).setOutputCol(outputCols).setCaseSensitive(False)
        return glove_embeddings

    def getBERTEmbeddings(self,inputCols,outputCols):
        """
        inputCols(list): documentCol, tokenCol
        outputCols(str)
        """
        bert_embeddings = BertEmbeddings.pretrained('bert_base_cased', 'en').setInputCols(inputCols).setOutputCol(outputCols).setCaseSensitive(False)
        return bert_embeddings

    def getELMOEmbeddings(self,inputCols,outputCols):
        """
        inputCols(list): documentCol, tokenCol
        outputCols(str)
        """
        elmo_embeddings = ElmoEmbeddings.pretrained().setPoolingLayer("word_emb").setInputCols(inputCols).setOutputCol(outputCols)
        return elmo_embeddings
        
    def getUSEEmbeddings(self,inputCols,outputCols):
        """
        inputCols(str): document
        outputCols(str)
        """
        useEmbeddings = UniversalSentenceEncoder.pretrained().setInputCols(inputCols).setOutputCol(outputCols)
        return useEmbeddings

    def getEmbeddingSentence(self,inputCols,outputCols):
        """
        inputCols(list): documentCol, embeddings
        outputCols(str)
        """
        embeddingsSentence = SentenceEmbeddings().setInputCols(inputCols).setOutputCol(outputCols).setPoolingStrategy("AVERAGE")
        return embeddingsSentence

    def getEmbeddingFinisher(self,inputCols,outputCols):
        """
        inputCols(str): sentenceEmbeddings
        outputCols(str)
        """
        embeddings_finisher = EmbeddingsFinisher().setInputCols([inputCols]).setOutputCols([outputCols]).setOutputAsVector(True).setCleanAnnotations(False) 
        return embeddings_finisher

    def getExplodeVectors(self,inputCols,outputCols):
        """
        inputCols(str)
        outputCols(str)
        """
        stmt = "SELECT EXPLODE("+inputCols+") AS "+outputCols+", * FROM __THIS__"
        explodeVectors = SQLTransformer(statement=stmt)
        return explodeVectors

    def getVectorAssembler(self,inputCols,outputCols):
        """
        inputCols(list):[inputcols]
        outputCols(str)
        """
        assembler = VectorAssembler(inputCols=inputCols,outputCol=outputCols)
        return assembler
    

    

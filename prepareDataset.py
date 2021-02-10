import os
import pandas as pd
import platform
import PyPDF2
import docx2txt
import glob
import textract
import pytesseract
from PIL import Image 
from pdf2image import convert_from_path 
from langdetect import detect
from googletrans import Translator
import re
#Sets path separator based on host OS
if platform.system()=='Windows':
    path_separator = "\""
else:
    path_separator = "/"

class prepareData:
    def read_files(self,parent_dir):
        """
        Summary:
            Reads all the files from folder recursively
        Parameters: 
            parent_dir(str): Path of parent directory
        Returns:
            list: all the files present in parent directory and its subdirectories 
        """
        try:
            subdirs = [x[0] for x in os.walk(parent_dir)]
            files = []
            for subdir in subdirs:
                files.extend(glob.glob(subdir+path_separator+"*.pdf"))
                files.extend(glob.glob(subdir+path_separator+"*.docx"))
                files.extend(glob.glob(subdir+path_separator+"*.doc"))
            return files
        except:
            print("An exception occured while reading data..")


    def read_ocr_file(self,file):
        """
        Summary:
            Extracts text from scanned pdf files using OCR 
        Parameters: 
            file(str): Path of pdf file
        Returns:
            str: text extracted from the file 
        """
        try:
            pages = convert_from_path(file, 500)
            text = ""
            for page in pages:  
                text += str(((pytesseract.image_to_string(page)))) 
            text = text.replace("\n","")
            return text
        except:
            print("An exception occured while reading pdf file using ocr..")


    def read_pdf_file(self,file):
        """
        Summary:
            Reads text from pdf files using PDFReader
            Directs to OCR Reader if file is scanned 
        Parameters: 
            file(str): Path of pdf file
        Returns:
            str: text extracted from the file 
        """
        try:
            pdfFileObj = open(file, 'rb') 
            pdfReader = PyPDF2.PdfFileReader(pdfFileObj,strict=False)
            pages = pdfReader.numPages
            text = ""
            for i in range(pages):
                pageObj = pdfReader.getPage(i)
                text+=pageObj.extractText()
            text = text.replace("\n","")
            text = text.replace("   ","")
            search_res = re.search('[a-zA-Z]', text)
            if not text or len(text.split(' '))<10 or not search_res:
                return self.read_ocr_file(file)
            return text
        except:
            print("An exception occured while reading pdf file..")

    def read_other_file(self,file):
        """
        Summary:
            Reads text from doc files 
        Parameters: 
            file(str): Path of doc file
        Returns:
            str: text extracted from the file 
        """
        try:
            text = textract.process(file)
            text = text.decode("utf-8") 
            text = text.replace("\n","")
            return text
        except:
            print("An exception occured while reading other file..")

    def read_docx_file(self,file):
        """
        Summary:
            Reads text from docx files 
        Parameters: 
            file(str): Path of docx file
        Returns:
            str: text extracted from the file 
        """
        try:
            text = docx2txt.process(file)
            text = text.replace("\n","")
            return text
        except:
            print("An exception occured while reading docx file..")

    def read_file_text(self,file):
        """
        Summary:
            Extracts the text from file
            File can be of pdf or docx type
        Parameters: 
            file(str): Complete path of file
        Returns:
            str: text extracted from the file 
        """
        try:
            if file.rsplit('.',1)[-1]=='pdf':
                return self.read_ocr_file(file)
            elif file.rsplit('.',1)[-1]=='docx':
                return self.read_docx_file(file)
            else:
                return self.read_other_file(file)
        except:
            print("An exception occured while reading data file..")

    def get_type(self,ext):
        try:
            if ext=='pdf':
                return 0.0
            elif ext=='docx':
                return 1.0
            else:
                return 2.0
        except:
            print("An exception occured while getting type..")

    def form_dataset(self,files,fileName):
        """
        Summary:
            Extracts the features from all the files in parent directory
        Parameters: 
            files(list): list of file paths to extract features
        Returns:
            pandas Dataframe: Dataframe containing all features   
        """
        try:
            df_cols = {'filename':[],'filepath':[],'filetype':[],'filesize':[],'filetext':[],\
                'translatedtext':[],'adj_nodes':[]}
            cnt = 0
            translator = Translator()
            for file in files:
                print(cnt,file)
                cnt+=1
                file_split = file.rsplit(path_separator,1)
                df_cols['filename'].append(file_split[-1])
                df_cols['filepath'].append([i for i in file_split[0].split(path_separator) if i])
                df_cols['filetype'].append(self.get_type(file.rsplit('.',1)[-1]))
                df_cols['filesize'].append(os.stat(file).st_size)
                df_cols['adj_nodes'].append(os.listdir(file_split[0]))
                file_text = self.read_file_text(file)
                df_cols['filetext'].append(file_text)
                if detect(file_text)=='en':
                    df_cols['translatedtext'].append(file_text)
                else:
                    tt = translator.translate(file_text)
                    df_cols['translatedtext'].append(tt.text)
            df = pd.DataFrame.from_dict(df_cols)
            df.to_csv(fileName,index=False)
            return df
        except:
            print("An exception occured while forming dataset..")
    

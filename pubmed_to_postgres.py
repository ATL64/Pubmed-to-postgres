#!/usr/bin/env python
# coding: utf-8

from collections import Counter
import csv
from datetime import datetime, timedelta
from google.cloud import storage
import io
import json
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import os
import pandas as pd
import psycopg2
import re
import requests
import sys
from typing import Iterator, Optional, Dict, Any, Tuple

# Custom module
import asn1
import mrc_stopwords

stopwords = mrc_stopwords.stopwords

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()

start_date = datetime(2020, 6, 2, 0, 0, 0)
end_date = datetime(2020, 9, 30, 0, 0, 0)
one_day = timedelta(days=1)
bucket_name = 'biotech_lee'
gcs_client = storage.Client()
bucket = gcs_client.bucket(bucket_name)

def connect_to_postgres_db():
    return psycopg2.connect("dbname='postgres' user='postgres' host='34.xx.xx.xx' port='5432' password='xxxxx'")

def get_pubmed_data_from_day(date):
    year, month, day = str(date.year), str(date.month), str(date.day)
    search_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&mindate='+year+'/'+month+'/'+day+'&maxdate='+year+'/'+month+'/'+day+'&usehistory=y&retmode=json'
    search_r = requests.post(search_url)
    search_data = search_r.json()
    webenv = search_data["esearchresult"]['webenv']
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmax=9999&query_key=1&webenv="+webenv
    with requests.post(fetch_url) as fetch_r:
        pubmed_data_string = fetch_r.content

    pubmed_data_string = pubmed_data_string.decode("utf-8")
    with open("Output.txt", "w") as text_file:
        text_file.write(pubmed_data_string)

    json_output = asn1.to_json("Output.txt")

    output_file = 'pubmed_json/'+year+'_'+month+'_'+day+'.json'
    blob = bucket.blob(output_file)
    blob.upload_from_string(json_output)

    os.remove('Output.txt')

    return json_output

# function to convert nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return 'a'
    elif nltk_tag.startswith('V'):
        return 'v'
    elif nltk_tag.startswith('N'):
        return 'n'
    elif nltk_tag.startswith('R'):
        return 'r'
    else:
        return None

def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    words = set([w.lower() for w in lemmatized_sentence if any(c.isalpha() for c in w)])
    words = [w for w in words if w not in stopwords]
    return words

def generate_mrc_data(json_data):
    output_items = json_data.split('\n')
    mrc_data = []
    keys_needed = ['pmid', 'abstract', 'article_title', 'article_date', 'journal', 'authors']
    for item in output_items:
        if len(item) == 0:
            continue
        try:
            obj = json.loads(item)
            data_dict = {}
            data_dict['pmid'] = obj['pmid']
            if 'abstract' not in obj['medent'].keys():
                continue
            data_dict['abstract'] = obj['medent']['abstract']
            if 'title' in obj['medent']['cit'].keys():
                data_dict['article_title'] = obj['medent']['cit']['title']['name']
            else:
                data_dict['article_title'] = '-'
            data_dict['article_date'] = obj['medent']['em_std']['year'] + '-' + obj['medent']['em_std']['month'] + '-' + obj['medent']['em_std']['day']
            if 'authors' in obj['medent']['cit'].keys():
                if 'names_std' in obj['medent']['cit']['authors'].keys():
                    data_dict['authors'] = ','.join([x['nameml'] for x in obj['medent']['cit']['authors']['names_std'] if 'nameml' in x.keys()])
                elif 'names_ml' in obj['medent']['cit']['authors'].keys():
                    data_dict['authors'] = ','.join(obj['medent']['cit']['authors']['names_ml'])
                else:
                    raise Exception('no author names')
            if 'from_journal' in obj['medent']['cit'].keys():
                data_dict['journal'] = obj['medent']['cit']['from_journal']['title']['iso_jta']
            else:
                data_dict['journal'] = obj['medent']['cit']['from_book']['title']['name']
                if 'authors' not in data_dict.keys():
                    if 'names_std' in obj['medent']['cit']['from_book']['authors'].keys():
                        for a in obj['medent']['cit']['from_book']['authors']['names_std']:
                            data_dict['authors'] = ','.join(list(a.values()))
                    elif 'names_ml' in obj['medent']['cit']['from_book']['authors'].keys():
                        data_dict['authors'] = ','.join(obj['medent']['cit']['from_book']['authors']['names_ml'])
                    else:
                        raise Exception('no author names')
            for k in keys_needed:
                if k not in data_dict.keys():
                    data_dict[k] = ''
            mrc_data.append(data_dict)
        except Exception as e:
            print(e)
            print(item, type(item), len(item))
            break

    for x in mrc_data:
        x['abstract'] = x['abstract'].replace('\\.', '').replace('\\xAE', '')

    keyword_counter = Counter()
    mrc_keywords = []
    for item in mrc_data:
        keywords = lemmatize_sentence(item['abstract'])
        for k in keywords:
            keyword_counter[k] += 1
            mrc_keywords.append((item['pmid'], k, int(item['article_date'][:4])))

    return mrc_data, mrc_keywords, keyword_counter

# Currently, we're only saving mrc_keywords. But now I'm going to save the keyword frequency as well
# We won't save the mrc_data for now, as it can 'easily' be obtained from the file saved in get_pubmed_data_from_day
# mrc_keywords is a list of lists. Every list is a row in the csv file
# mrc_keywords is a dict k:v should be saved as k,v in a line of the csv file
def save_data_to_gcs(mrc_data, mrc_keywords, keyword_freq, date):
    year, month, day = str(date.year), str(date.month), str(date.day)

    # Save mrc_keywords
    with open("Output.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(mrc_keywords)
    output_file = 'mrc_keywords/'+year+'_'+month+'_'+day+'.csv'
    blob = bucket.blob(output_file)
    blob.upload_from_filename('Output.csv')
    os.remove('Output.csv')

    # Save keyword frequency
    kf = []
    for k,v in keyword_freq.items():
        kf.append([k,v])
    with open("Output.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(kf)
    output_file = 'mrc_keyword_freq_new/'+year+'_'+month+'_'+day+'.csv'
    blob = bucket.blob(output_file)
    blob.upload_from_filename('Output.csv')
    os.remove('Output.csv')
    return

# Class to upload data to PostgreSQL without the need of using a csv or JSON file
class StringIteratorIO(io.TextIOBase):
    def __init__(self, iter: Iterator[str]):
        self._iter = iter
        self._left = ''

    def readable(self) -> bool:
        return True

    def _read1(self, n: Optional[int] = None) -> str:
        while not self._left:
            try:
                self._left = next(self._iter)
            except StopIteration:
                break
        ret = self._left[:n]
        self._left = self._left[len(ret):]
        return ret

    def read(self, n: Optional[int] = None) -> str:
        line = []
        if n is None or n < 0:
            while True:
                m = self._read1()
                if not m:
                    break
                line.append(m)
        else:
            while n > 0:
                m = self._read1(n)
                if not m:
                    break
                n -= len(m)
                line.append(m)
        return ''.join(line)

    def readline(self):
        l = []
        while True:
            i = self._left.find('\n')
            if i == -1:
                l.append(self._left)
                try:
                    self._left = next(self._iter)
                except StopIteration:
                    self._left = ''
                    break
            else:
                l.append(self._left[:i+1])
                self._left = self._left[i+1:]
                break
        return ''.join(l)

def clean_csv_value(value: Optional[Any]) -> str:
    if value is None:
        return r'\N'
    return str(value).replace('\n', '\\n')

def copy_string_iterator_mrc_data(connection, table_name, data: Iterator[Dict[str, Any]], size: int = 8192):
    with connection.cursor() as cursor:
        string_iterator = StringIteratorIO((
            '\t'.join(map(clean_csv_value, (
                x['pmid'],
                x['abstract'],
                x['article_title'],
                x['article_date'],
                x['journal'],
                x['authors']
            ))) + '\n'
            for x in data
        ))
        cursor.copy_from(string_iterator, table_name, sep='\t', size=size)
    return

def upload_mrc_data_to_postgres(mrc_data):
    conn = connect_to_postgres_db()
    copy_string_iterator_mrc_data(conn, 'prod.mrc_data', mrc_data)
    conn.commit()
    conn.close()
    return

def copy_string_iterator_mrc_keywords(connection, table_name, data: Iterator[Tuple[str, str, str]], size: int = 8192):
    with connection.cursor() as cursor:
        string_iterator = StringIteratorIO((
            '\t'.join(map(clean_csv_value, (
                x[0],
                x[1],
                x[2]
            ))) + '\n'
            for x in data
        ))
        cursor.copy_from(string_iterator, table_name, sep='\t', size=size)
    return

def upload_mrc_keywords_to_postgres(mrc_keywords):
    conn = connect_to_postgres_db()
    copy_string_iterator_mrc_keywords(conn, 'prod.mrc_keywords', mrc_keywords)
    conn.commit()
    conn.close()
    return

date = start_date
while date <= end_date:
    print(date)
    day_data = get_pubmed_data_from_day(date)
    print('Data from PubMed successfully read.')
    mrc_data, mrc_keywords, keyword_freq = generate_mrc_data(day_data)
    print('MRC data successfully generated')
    save_data_to_gcs(mrc_data, mrc_keywords, keyword_freq, date)
    print('Data successfully uploaded to GCS')
    upload_mrc_data_to_postgres(mrc_data)
    print('mrc_data successfully uploaded to Postgres')
    upload_mrc_keywords_to_postgres(mrc_keywords)
    print('mrc_keywords successfully uploaded to Postgres')

    date += one_day

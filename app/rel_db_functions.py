from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import csv
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from collections import defaultdict
import time

db = SQLAlchemy()



def bm25_search(query, top_n=10):
    docs = []

    start = time.time()
    for material in Material.query.all():
        ita = material.short_desc_it.strip()
        eng = material.short_desc_eng.strip()

        if material.long_desc_it:
            ita = ita + ' ' + material.long_desc_it.strip()
        if material.long_desc_eng:
            eng = eng + ' ' + material.long_desc_eng.strip()
        docs.append((material.id, ita, ''))
        docs.append((material.id, eng, ''))
    end = time.time()
    print(f"Time taken to fetch documents: {end - start} seconds")

    start = time.time()
    tokenized_docs = [word_tokenize(doc[1].lower()) for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    end = time.time()
    print(f"Time taken to calculate scores: {end - start} seconds")

    start = time.time()
    for i, score in enumerate(scores):
        docs[i] = (docs[i][0], docs[i][1], score)
    docs = sorted(docs, key=lambda x: x[2], reverse=True)
    end = time.time()
    print(f"Time taken to sort documents: {end - start} seconds")

    start = time.time()
    results = []
    for d in docs:
        if d[0] not in [r[0] for r in results]:
            results.append((d[0], Material.query.filter_by(id=d[0]).first().short_desc_it, Material.query.filter_by(id=d[0]).first().short_desc_eng, d[2]))
        if len(results) >= top_n:
            break
    end = time.time()
    print(f"Time taken to filter results: {end - start} seconds")
    return results

class Material(db.Model):
    id = db.Column(db.String, primary_key=True)
    short_desc_it = db.Column(db.String, primary_key=False)
    short_desc_eng = db.Column(db.String, primary_key=False)
    long_desc_it = db.Column(db.String, primary_key=False)
    long_desc_eng = db.Column(db.String, primary_key=False)

def add_materials():
    with open ("../datasets/catalogue_01_part_1.csv") as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)
        for row in reader:
            material = Material(
                id=row[0],
                short_desc_it=row[1],
                short_desc_eng=row[3],
                long_desc_it=None,
                long_desc_eng=None,
            )
            db.session.add(material)
        db.session.commit()
        
    with open ("../datasets/catalogue_01_part_2.csv") as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)
        for row in reader:
            material = Material(
                id=row[0],
                short_desc_it=row[1],
                short_desc_eng=row[3],
                long_desc_it=None,
                long_desc_eng=None,
            )
            db.session.add(material)
        db.session.commit()

def add_long_description():
    print("Starting to add long descriptions in Italian")
    with open("../glossario_ita.csv") as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)
        long_desc = ''
        for row in reader:
            for material in Material.query.all():
                if material.long_desc_it is None:
                    long_desc = material.short_desc_it.strip()
                    # print("Long desc is None")
                else:
                    long_desc = material.long_desc_it
                if (row[0].strip() + ' ') in long_desc:
                    long_desc = long_desc.replace(row[0].strip(), row[1].strip())
                    material.long_desc_it = long_desc
                    db.session.commit()
    print("Finished adding long descriptions in Italian")

    print("Starting to add long descriptions in English")
    with open("../glossario_eng.csv") as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)
        long_desc = ''
        for row in reader:
            for material in Material.query.all():
                if material.long_desc_eng is None:
                    long_desc = material.short_desc_eng.strip()
                else:
                    long_desc = material.long_desc_eng
                if (row[0].strip() + ' ') in long_desc:
                    long_desc = long_desc.replace(row[0].strip(), row[1].strip())
                    material.long_desc_eng = long_desc
                    db.session.commit()
    print("Finished adding long descriptions in English")

def reset_db():
    db.drop_all()
    db.create_all()
    print("Database resettato.")
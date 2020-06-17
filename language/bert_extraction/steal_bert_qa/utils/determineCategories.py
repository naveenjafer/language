import spacy
import json
import os
import textacy
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree

import tensorflow.compat.v1 as tf
flags = tf.flags

flags.DEFINE_string("input_path", None, "Path containing the SQuAD training data.")
flags.DEFINE_string("output_path", None, "Path where the modified squad dataset training data with the categories needs to be written to.")
flags.DEFINE_string("corenlp_path", None, "Path where the core nlp files are located")
FLAGS = flags.FLAGS

# /home/naveen/scratch/google-language-fork/stanford-corenlp-full-2018-10-05
nlpS = StanfordCoreNLP(FLAGS.corenlp_path)
nlp = spacy.load("en_core_web_sm")
record = {"names" : 0, "numbers" : 0, "places" : 0, "dates" : 0, "otherEnts" : 0, "noun_phrases" : 0, "verb_phrases" : 0, "adjective_phrases" : 0, "clauses" : 0, "others" : 0}
recordStartingWord = {
    "names": {},
    "numbers": {},
    "places": {},
    "dates": {},
    "otherEnts": {},
    "noun_phrases": {},
    "verb_phrases": {},
    "adjective_phrases": {},
    "clauses": {},
    "others": {}
}
for key in recordStartingWord:
    recordStartingWord[key]["total"] = 0
    recordStartingWord[key]["words"] = {}
test = []
complexNP = 0
complexNPList = []
totalSum = 0

def readDataFile(fileName):
    global record
    global totalSum
    global test
    with open(fileName) as f:
        data = json.load(f)

    for titleItem in data["data"]:
        for paragraphs in titleItem["paragraphs"]:
            #print("Processing Para ", index)
            for contextItem in paragraphs["qas"]:
                answerItem = contextItem["answers"]
                question = contextItem["question"]
                questionList = question.split(" ")
                firstWordInQuestion = ""
                if len(questionList) > 0:
                    firstWordInQuestion = questionList[0]
                if len(answerItem) > 0:
                    #print(answerItem)
                    text = answerItem[0]["text"]
                    category = processText(text)
                    if firstWordInQuestion.lower() in recordStartingWord[category]["words"]:
                        recordStartingWord[category]["words"][firstWordInQuestion.lower()] = recordStartingWord[category]["words"][firstWordInQuestion.lower()] + 1
                    else:
                        recordStartingWord[category]["words"][firstWordInQuestion.lower()] = 1
                    recordStartingWord[category]["total"] = recordStartingWord[category]["total"] + 1
                    contextItem["category"] = category

                totalSum = totalSum + 1
    with open(FLAGS.output_path, "w") as outfile:
        json.dump(data, outfile)
    #print(totalSum)
    #print(record)

def cleanText(text):
    if text != "":
        try:
            if text[len(text)-1] == ".":
                text = text[0:len(text)-1]
            if text[0] == "\"" or text[0] == '\'':
                text = text[1:]
            if text[len(text)-1] == "\"":
                text = text[0:len(text)-1]
            text = text.strip()
        except:
            print("The text that crashed is:",text,"EOF")
        return text

def processText(text):
    global record
    global test
    global complexNP
    doc = nlp(text)
    allNames = False
    allPlaces = False
    allDates = False
    allNumbers = False
    allOtherEnt = False
    text = cleanText(text)
    # the resource to all other entity types as in SpaCy - https://github.com/explosion/spaCy/issues/441#issuecomment-311804705
    for ent in doc.ents:
        allNames = True
        allPlaces = True
        allDates = True
        allNumbers = True
        if ent.label_ != "DATE":
            allDates = False
        if ent.label_ != "PERSON":
            allNames = False
        if ent.label_ != "GPE":
            allPlaces = False
        if ent.label_ != "CARDINAL" and ent.label_ != "PERCENT" and ent.label_ != "MONEY":
            allNumbers = False

    if allNames == True:
        record["names"] = record["names"] + 1
        return "names"
        allOtherEnt = True

    if allPlaces == True:
        record["places"] = record["places"] + 1
        return "places"
        allOtherEnt = True

    if allDates == True:
        record["dates"] = record["dates"] + 1
        return "dates"
        allOtherEnt = True

    if allNumbers == True:
        record["numbers"] = record["numbers"] + 1
        return "numbers"
        allOtherEnt = True

    if allOtherEnt == False:
        if len(doc.ents) > 0:
            lengthOfEnts = 0
            for ent in doc.ents:
                lengthOfEnts = lengthOfEnts + ent.end_char - ent.start_char
            if lengthOfEnts == len(text):
                record["otherEnts"] = record["otherEnts"] + 1
                return "otherEnts"
                test.append(text)
                allOtherEnt = True

    if allOtherEnt == False:
        #now we start checking for the phrases.
        lengthOfNP = 0
        NPFlag = False
        for np in doc.noun_chunks:
            lengthOfNP = lengthOfNP + np.end_char - np.start_char
        if lengthOfNP == len(text):
            record["noun_phrases"] = record["noun_phrases"] + 1
            return "noun_phrases"
            NPFlag = True
        else:
            complexNP = complexNP + 1
            #complexNPList.append(text)
        if NPFlag == True:
            return
        #isVerbPhrase(text)
        if isNounPhraseCoreNLP(text) == True:
            return "noun_phrases"

        if isVerbPhraseCoreNLP(text) == True:
            return "verb_phrases"

        if isAdjPhraseCoreNLP(text) == True:
            return "adjective_phrases"

        if isClauseCoreNLP(text) == True:
            return "clauses"

        record["others"] = record["others"] + 1
        complexNPList.append(text)
        return "others"


def isVerbPhrase(text):
    global record
    #pattern = r'(<VERB>?<ADV>*<VERB>+)'
    pattern = [{"POS" : "VERB", "OP" : "?"}, {"POS" : "ADV", "OP" : "*"}, {"POS" : "VERB", "OP" : "+"}]
    doc = textacy.make_spacy_doc(text, lang='en_core_web_sm')
    verb_phrases = textacy.extract.matches(doc, pattern)
    lengthOfVP = 0
    for chunk in verb_phrases:
        #print(chunk.text)
        lengthOfVP = lengthOfVP + chunk.end_char - chunk.start_char
    if lengthOfVP == len(text):
        record["verb_phrases"] = record["verb_phrases"] + 1
    return True

def extract_phrase(tree_str, label):
    phrases = []
    trees = Tree.fromstring(tree_str)
    for tree in trees:
        for subtree in tree.subtrees():
            if subtree.label() == label:
                t = subtree
                t = ' '.join(t.leaves())
                phrases.append(t)
    return phrases

def isVerbPhraseCoreNLP(text):
    tree_str = nlpS.parse(text)
    vps = extract_phrase(tree_str, "VP")
    for item in vps:
        if item == text:
            record["verb_phrases"] = record["verb_phrases"] + 1
            return True
    return False

def isNounPhraseCoreNLP(text):
    tree_str = nlpS.parse(text)
    #print(tree_str)
    nps = extract_phrase(tree_str, "NP")
    #print(nps)
    for item in nps:
        if item == text:
            record["noun_phrases"] = record["noun_phrases"] + 1
            return True
    return False

def isAdjPhraseCoreNLP(text):
    tree_str = nlpS.parse(text)
    adjp = extract_phrase(tree_str, "ADJP")
    for item in adjp:
        if item == text:
            record["adjective_phrases"] = record["adjective_phrases"] + 1
            return True
    return False

def isClauseCoreNLP(text):
    tree_str = nlpS.parse(text)
    cp = extract_phrase(tree_str, "S")
    for item in cp:
        if item == text:
            record["clauses"] = record["clauses"] + 1
            return True
    return False

if __name__ == "__main__":
    readDataFile(FLAGS.input_path)
    for item in record:
        print(item, ": ", record[item]*100/totalSum)
        firstWords = recordStartingWord[item]["words"]
        firstWordsSorted = sorted(firstWords.items(), key = lambda x: x[1], reverse=True)
        counter = 0
        print("The top 3 words are")
        print(firstWordsSorted[0:3])
        for itemInner in firstWordsSorted:
            counter = counter + 1
            print("\t", itemInner[0], " \t", itemInner[1]/recordStartingWord[item]["total"])

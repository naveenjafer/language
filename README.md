# Google AI Language
This is a clone of the https://ai.google/research/teams/language/ repo from the google open source projects.

# Instructions
## Clone this repo
Clone this repo and switch to the branch "colabCompatible"

## Virtual env
Create a virtual env
`python3 -m venv env`

Activate the virtual env
`source env/bin/activate`

## Python Packages
```
pip install tensorflow-gpu~=1.15.0
pip install bert-tensorflow
pip install tqdm
pip install spacy
pip install textacy
pip install nltk
pip install stanfordcorenlp
python -m spacy download en_core_web_sm
```

## Build and install
```
cd language
python3 setup.py build && python3 setup.py install
```

## Modify Permissions
Since we are concerned with the extraction of squad in this repo, set permissions to the following shell scripts.  
`chmod +x language/bert_extraction/steal_bert_qa/scripts/*`

## Prepare the folders needed
Decide on a location where all the relevant generated folders and files will be placed/created. I have picked the level where this repo has been duplicated for generating my folders.
The cloned folder is named "language" so I have placed all the other folders created inside one folder "generatedFolders" that is at the same level as "language".  
`cd ..`

### Download Bert model.
```
mkdir generatedFolders
cd generatedFolders
mkdir bertModelVictim
mkdir bertModelExtracted
cd bertModelVictim
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
cp -rf uncased_L-12_H-768_A-12 ../bertModelExtracted/
cd ..
```

### Create other folders
```
mkdir squadDir
mkdir outputDirVictim
mkdit outputDirExtracted
mkdir extractionDir
mkdir wikiDir
cd wikiDir
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
cd ..
```

Download the corenlp folder from here https://drive.google.com/file/d/1t2-jFJeDwCyQTBU11PY5ayvMZC2z-3qy/view?usp=sharing
Ensure that it is named as "stanford-corenlp-full-2018-10-05" and place it in the same level as the other folders created above.

## Running the code
### Script 1
`cd language/language/bert_extraction/steal_bert_qa/scripts/`  

This script trains the victim model and outputs the classwise F1 and exact match scores.  

`./train_victim_squad.sh <arg1> <arg2> <arg3> <arg4>`  

arg1 - Full path to the uncased_L-12_H-768_A-12 folder inside bertModelVictim above  
arg2 - Full Path to squadDir folder created above  
arg3 - Full path to outputDirVictim folder created above  
arg4 - Full path to stanford-corenlp-full-2018-10-05 folder created above  

### Script 2
Todo

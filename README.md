# Humor Generation from Augmented News Headlines



## Requirements

* python 3.6
* Pytorch >= 1.0
* transformers
* Stanza



## Pipeline Structure

1. **Preprocessing**: From the original dataset, identify "object" & "oblique nominal" nouns, as well as the verbs which they are associated with.

Please find the code to perform preprocessing within the "Preprocessing.ipynb" file.

* This notebook should utilize stanza to parse the dataset we want to augment - the initial modified news headlines dataset from the paper ("President Vows to Cut \<Taxes> Hair", by Hossain et al.)  is included in the file "datasets/news-headlines-humor"

  

2. **Word Context Replacement**: Train a CBOW model to identify alternatives to the target object.

Please find code to train and utilize the continuous bag of words model in the  "CBOW.ipynb" file.

* The notebook should utilize the "News category dataset" from kaggle: https://www.kaggle.com/rmisra/news-category-dataset, which should be included in the "datasets" folder.



3. **Scoring and Filtering**: After the word replacements, filter out the outputs that score low on grammaticality, and prefer those that have high "surprisal ratios".

Note: In order to perform scoring, we will need to utilize the github repository for the paper "Pun Generation with Surprise", by He et al. (https://github.com/hhexiy/pungen). This repository should be included under "pungen-master"

*  This repository also requires use of the repository "fairseq" in order to perform the scoring. That code will need to be installed as well, please find instructions in the the "pungen" git repository Readme.
* Upon completing the above steps, run the scoring and filtering with:

```bash
python eval_filter_surprisal_altered_hdls.py\
 --from-file True\
 --sents-csv-path [path/to/step1/output]\
 --outfile [path/to/scored/output]
```

* (note the above file is located in pungen_master/)
* After scoring, run the file "pungen_master/filter_grammar_humor_scores.py". Before running, set the parameters:
  *  "input_path" = scored output from the last run
  * "output_path" = store the output in your workspace



4. **Silly Synonyms**: Find "sillier" words to use, instead of the object and associated verb.

Please find code to perform the silly synonym replacement in the notebook "silly_synonyms.ipynb".

* Requirements:
  * You will need to download Glove Embeddings
  * You will also need the "funny words" dataset from the paper "Humor Norms for 4,997 English Words", by Engelthaler and Hill. This should be included in the "datasets" folder.
* You will need to run the "silly synonym code" section, as well as the "silly synonym inference script". In the final cell, set the following variables:
  *  "input path" = output from the last step
  * "Output_path" = store the output in your workspace
    

5. **Score Again**: If there are a large number of outputs generated, it may be beneficial to re-score the outputs.

A script to score the outputs from the final step is included at "pungen-master/eval_filter_surprisal_scores.py", it should take the same arguments as step 3.1.

For BERT-based evaluation, of the outputs, please see the notebook "Bert_humor_scoring". You will need to run the Preprocessing, training, and humor analysis sections.
* Requirements: the data we used to train BERT is included in datasets/bert_training/, and some required preprocessing scripts should be included in the "helpers" folder.
* please modify the parameter "fpath1" in the second cell of the "analysis" section to the directory of your model outputs.


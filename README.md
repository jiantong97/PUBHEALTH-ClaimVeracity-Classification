# PUBHEALTH-ClaimVeracity-Classification
## Problem Definition
The main task is to predict the label('true', 'false', 'unproven', 'mixture') for veracity of public health claim based on main text which is the source of the claim. The problem is a multi-class sentence pair classification task.

## Outline
1. Data Preprocessing
    1. Drop missing value
    2. Remove outliers for labels
    3. check whether imbalanced class
2. Data Transformation--Fetch top k similar sentences from main text--Sentence BERT
3. Train various transformer_based models on train-set and tune hyper-parameters on dev-set
    1. BERT
       * with all sentences in main text
       * with topk sentences in main texts--cross entropy loss
       * with topk sentences in main text--focal loss
    2. SciBERT--Pre_trained on specific domain
    3. ALBERT
    4. BigBird
    5. DeBERTaV3
4. Evaluate on test set and final results in terms of macro F1, Precision, Recalland accuracy 

## Note
For part 1 and 2, I run on notebook for the interactive display of each steps and I also make it a .py scripts for consistent data preprocessing.

For part 2 and 3, I run on Google Colab to utilize the GPU and High-RAM computational resources on Google Cloud.

For part 4, I run on notebook for the interactive display of each model.

The fine_tuned transformer_based models are stored in the fine_tuned_model folder. Datas are in the Data folder. Prediction results are stored in results folder.

## Code files for each stop
1. Data Preprocessing
    * Notebooks/1.Data Preprocessing.ipynb
    * Scripts/1.Data_preprocessing.py
2. Fetch top k similar sentences from main text--Sentence BERT
    * Notebooks/2-3-4.Veracity Classification.ipynb--fetch k top sentences from main text
    * Scripts/2.fetch_top_k_sentences.py
3. Train and validate
    * Notebooks/2-3-4.Veracity Classification.ipynb
4. Evaluate
    * Notebooks/2-3-4.Veracity Classification.ipynb--Evaluate

## Detailed Description
**1. Data Preprocessing**
* Fristly I get the columns needed for prediction and find that origin data exists some missing value for main text, so I drop the missing value.
* Secondly I find there exists 5 labels for train and dev data, so I drop the data with outlier labels.
* Finally I draw the distribution of four labels on each set, the distributions are same with three sets, but the four labels are imbalanced class with less 'mixture' and 'unproven' labels. So I should use metrics like macro F1 to evaluate and I also tried weighted loss like focal loss but it didn't improve the performance significantly.
* The cleaned version of data is stored as clean_xxx.tsv.

**2. Fetch top k sentences**
* Since the main text is too long that exceeds 512 tokens for certain. A better way is to extract useful texts from the main text.
* I firstly tokenize main text into sentences and apply pre_trained Sentence BERT model which can return the similarity of claim and each sentence of main text.
* Finally I fetch the top k similar sentences and make it a new column called topk_text. I set k as 5 from the fact-checking literature (Nie et al., 2019; Zhong et al., 2019)
* I run on Google Colab for this step. And I only need to pip install sentence_transformers in colab environment.
* The processed data is stored as clean_xxx_topk.tsv

**3.Train and validate**
* This is a sentence-pair classification problem. I adopted several pre_trained transformers_based models in Hugging Face with Pytorch and Transformers package. 
* I run on Google Colab and I only need to pip install transformers in colab environment.
* There are four types of transformes_based models that I used.
    * BERT. I fine_tuned 'bert-base-uncased' with all sentences in main_text, top k sentences in main text with cross entropy loss and focal loss. The result shows top k sentences extraction improves performance but change of loss fucntion doesn't.
    * BERT pre_trained on scientific corpus--SciBERT('allenai/scibert_scivocab_uncased').
    * Other encoder BERT with different attention implementation or pre_trained tasks and etc. I choose ALBERT('albert-base-v2') and BigBird('google/bigbird-roberta-base').
    * BERT with encoder and decorder. DeBERTaV3('microsoft/deberta-v3-base')
* The hyper parameters I choose follow (Neema et al., 2020) with learning rate = 1e-6, optimizer=AdamW, epoch=4, batch_size=16(8 for BigBird and DeBERTaV3 due to cuda memory limitation)

**4.Evaluation and results**
* I evaluate the prediction using metrics of macro-F1, precision, recall and accuracy. 
* The final results in shown in the follwing table. My baseline is bert-base-uncased.

Model|Macro Precision|Macro Recall|Marco F1|Accuracy
---|:--:|:--:|:--:|---:|
BERT(all sentences)(Baseline)|0.26|0.31|0.27|0.54
BERT(topk sentences with ce loss)|0.28|0.35|0.31|0.58
BERT(topk sentences with focal loss)|0.28|0.34|0.31|0.58
SciBERT(topk sentences with ce loss)|0.31|0.39|0.34|0.62
ALBERT(topk sentences with ce loss)|0.47|**0.42**|**0.38**|**0.65**
BigBird(topk sentences with ce loss)|**0.54**|0.36|0.32|0.58
DeBERTaV3(topk sentences with ce loss)|0.28|0.34|0.30|0.57

* From the results, we can see ALBERT outperforms other models with 0.38 macro F1 and 0.65 accuracy but all the models lack the ability to solve class imbalanced problem.

**5.Future work**

Due to the time and computational resouces limit, there are several ways that may improve performance and solve the imbalanced class problem but I didn't try 

* Firstly, I can try to under/over sample data to balance the class.
* Besides, I can try more transformer_based models and finetune their hyper-parameters like learning rate case by case.

import nltk
import heapq
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# function to get top k similar sentences
def get_top_k_sentences_text(claim, main, k):
    claim_representation = model.encode(claim, convert_to_tensor=True)
    main_representation = model.encode(main, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(claim_representation, main_representation).squeeze(0).tolist()
    topk = heapq.nlargest(5, range(len(similarity)), similarity.__getitem__)
    output = ''
    for i in topk:
      output += main[i] + " "
    return  output

if __name__ == '__main__'
    nltk.download('punkt')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #read data
    df_train = pd.read_csv('../Data/clean_train.tsv', sep='\t')
    df_val =  pd.read_csv('../Data/clean_dev.tsv', sep='\t')
    df_test =  pd.read_csv('../Data/clean_test.tsv', sep='\t')
    df_train['sentences'] = df_train.apply(lambda x: tokenizer.tokenize(x['main_text']), axis=1)
    df_val['sentences'] = df_val.apply(lambda x: tokenizer.tokenize(x['main_text']), axis=1)
    df_test['sentences'] = df_test.apply(lambda x: tokenizer.tokenize(x['main_text']), axis=1)

    #read sentence transformer model
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    #get top k similar sentences
    df_train['topk_text'] = df_train.apply(lambda x: get_top_k_sentences_text(x['claim'], x['sentences'], 5), axis=1)
    df_val['topk_text'] = df_val.apply(lambda x: get_top_k_sentences_text(x['claim'], x['sentences'], 5), axis=1)
    df_test['topk_text'] = df_test.apply(lambda x: get_top_k_sentences_text(x['claim'], x['sentences'], 5), axis=1)
    #save data
    df_train.to_csv('../Data/clean_train_topk.tsv', sep='\t', index=False)
    df_val.to_csv('../Data/clean_dev_topk.tsv', sep='\t', index=False)
    df_test.to_csv('../Data/clean_test_topk.tsv', sep='\t', index=False)
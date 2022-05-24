import pandas as pd


if __name__ == '__main__':
    train = pd.read_csv('../Data/train.tsv', sep='\t')
    dev = pd.read_csv('../Data/dev.tsv', sep='\t')
    test = pd.read_csv('../Data/test.tsv', sep='\t',index_col=0)

    #get columns needed for modeling
    train_df = train[['label', 'claim','main_text']]
    dev_df = dev[['label', 'claim','main_text']]
    test_df = test[['label', 'claim','main_text']]

    #1. Drop missing value
    train_df = train_df.dropna()
    dev_df = dev_df.dropna()
    test_df = test_df.dropna()

    #2. Only keep records with correct labels
    train_df = train_df[(train_df.label == 'unproven') | (train_df.label == 'true') |
                        (train_df.label == 'mixture') | (train_df.label == 'false')]
    dev_df = dev_df[(dev_df.label == 'unproven') | (dev_df.label == 'true') |
                        (dev_df.label == 'mixture') | (dev_df.label == 'false')]

    # save the clean data
    train_df.to_csv('../Data/clean_train.tsv', sep='\t', index=False)
    dev_df.to_csv('../Data/clean_dev.tsv', sep='\t', index=False)
    test_df.to_csv('../Data/clean_test.tsv', sep='\t', index=False)
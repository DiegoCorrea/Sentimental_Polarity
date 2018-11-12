import pandas as pd


def get_dataset_movie_sentiment_as_df():
    '''
    Polarity is either 1 (for positive) or 0 (for negative)
    :return: Um dataframe com os três datasets
    '''
    return pd.concat([
        pd.read_csv('datasets/raw_data/Movie_Sentiment_Analysis/amazon_cells_labelled.txt', sep='\t',
                    names=['sentence', 'polarity']),
        pd.read_csv('datasets/raw_data/Movie_Sentiment_Analysis/imdb_labelled.txt', sep='\t',
                    names=['sentence', 'polarity']),
        pd.read_csv('datasets/raw_data/Movie_Sentiment_Analysis/yelp_labelled.txt', sep='\t',
                    names=['sentence', 'polarity'])
    ], sort=False, ignore_index=True)


def get_dataset_polarity_as_df():
    '''
    Polarity is either 1 (for positive) or 0 (for negative)
    :return: Um dataframe com o dataset
    '''
    dataset = pd.read_json('datasets/raw_data/Polarity/examples.json')
    return dataset.rename(columns={'label': 'polarity', 'text': 'sentence'})


def get_dataset_sentence_polarity_as_df():
    '''
    Polarity is either 1 (for positive) or 0 (for negative)
    :return: Um dataframe com o dataset
    '''
    pos_df = pd.read_csv('datasets/raw_data/Sentence_Polarity_Dataset/rt-polarity.pos', sep='\n', names=['sentence'])
    pos_df['polarity'] = 1
    neg_df = pd.read_csv('datasets/raw_data/Sentence_Polarity_Dataset/rt-polarity.neg', sep='\n', names=['sentence'])
    neg_df['polarity'] = 0
    return pd.concat([pos_df, neg_df], sort=False, ignore_index=True)


def make_dataset():
    print("Construct a new dataset")
    dataset = pd.concat(
        [get_dataset_movie_sentiment_as_df(), get_dataset_polarity_as_df(), get_dataset_sentence_polarity_as_df()],
        sort=False,
        ignore_index=True
    )
    # print(dataset)
    dataset.to_csv('datasets/clean_data/polarity_dataset_concat.csv', sep='\t')
    print(pd.value_counts(dataset['polarity'].values, sort=False))
    return dataset

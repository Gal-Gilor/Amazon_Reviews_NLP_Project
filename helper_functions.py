import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist, word_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import recall_score, accuracy_score, roc_curve, auc, confusion_matrix, roc_auc_score, f1_score
import wordcloud
from wordcloud import WordCloud
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
import string
import itertools
from tqdm import tqdm


def remove_reviews(df, column_name, rating_tuple):
    """
    remove_reviews(df, column_name, rating_tuple)
    Params:
        df
        column_name: name of the column you want to remove
        rating_tuple: tuple containing the rating and the amount of observations
                      you wish to drop
    Returns:
        This function randomly removes observations and returns a new dataframe 
    """
    # extract observations with the same ratings
    same_ratings_df = df.loc[df[column_name] == rating_tuple[0]]
    
    # randomly drop observations from the dataset
    drop_indices = np.random.choice(same_ratings_df.index, rating_tuple[1], replace=False)
    df_final = df.drop(drop_indices)
    return df_final


def clean_review(review):
    """
    clean_review(review):
    Returns the text of a review without puncuation and capital letters
    Params:
        review: individual review from Amazon Product Review dataset
    Returns:
        a review with only lowercase letters and with puncuation removed
    """
    clean = []
    joined_clean_review = ''
    # for each element in the review
    for x in review:
        # if the element is a punctuation, replace it with a space
        if x in string.punctuation:
            x = x.replace(x, " ")
        # otherwise turn the letter into its lowercase form    
        elif x not in string.punctuation:
            x = x.lower()
        # append the letter to the empty list
        clean.append(x)    
        # join the letters into words
        joined_clean_review = "".join(clean)

    return joined_clean_review


def get_tokens(clean_review, final_stopwords):
    
    """ 
    get_tokens(clean_review):
    Returns a list of the individual from a review
    Params:
        clean_review: a review that has been wiped of its puncuation and capital letters
    Returns:
        a list of words, excluding "stop words", that comprise a review
    """
    
    #  tokenize & remove stop words
    list_of_tokens = [x for x in word_tokenize(clean_review) if x not in final_stopwords]
    
    # return list of text from each review
    return list_of_tokens


def lem_words(list_of_tokens, lemmatizer):
    """
    lem_words(list_of_tokens, lemmatizer):
    Returns the lemmas of each token
    Params:
        list_of_tokens: list of words (tokens) from a single review
        lemmatizer: instance of the NLTK lemmatizer class
    Returns:
        a string of lemmas that comprise a review
    """
    wrd_list = [lemmatizer.lemmatize(word) for word in list_of_tokens]
    # join the individual lemmas into a single string
    return " ".join(wrd_list)


def finalize_token(reviews, final_stopwords, lemmatizer):
    """
    finalize_token(reviews):
    Returns the final corpus of reviews
    Params:
        reviews: reviews that have been cleaned
    Returns:
        A list where each element of the list is a string representing a "cleaned" review
    """
    corpus = []
    for review in tqdm(reviews):
        clean = clean_review(review)
        tokens = get_tokens(clean, final_stopwords)
        lemmas = lem_words(tokens, lemmatizer)
        corpus.append(lemmas)
    return corpus


def CorrMtx(corr_map_df, dropDuplicates=True, xrot=70, label='Variable', save=False):
    """
    CorrMtx(corr_map_df, dropDuplicates=True, xrot=70, yrot=0, label='Variable')
    Params:
        corr_map_df: Pandas correlation map
        dropDuplicates: bool, default = True. Using masking to remove top right side of image
        xrot: int, default = 70. Assumes some columns have longer names
        label: string, default = 'Variable'. The x/ylabel title names.
    Returns:
        Returns a correlation heat map
    """
    # exclude duplicate correlations by masking uper right values
    if dropDuplicates:
        mask = np.zeros_like(corr_map_df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    # set background color / chart style
    sns.set_style(style='white')
    fig, ax = plt.subplots(figsize=(12, 10))

    # add diverging colormap from red to blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    # add titles
    plt.title("Correlation Heat Map")

    # draw correlation plot with or without duplicates
    if dropDuplicates:
        sns.heatmap(corr_map_df, mask=mask, cmap=cmap,
                    square=True,
                    linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
        plt.xlabel(label)
        plt.ylabel(label)
        plt.xticks(rotation=xrot)

    else:
        sns.heatmap(corr_map_df, cmap=cmap,
                    square=True,
                    linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
        plt.xlabel(label)
        plt.ylabel(label)
        plt.xticks(rotation=xrot)
    
    if save:
        plt.tight_layout()
        plt.savefight('cm_heatmap.png')
    return


def plot_confusion_matrix(y_test, y_pred, class_names, save=False, name='name'):
    """
    plot_confusion_matrix(y_test, y_pred, class_names, save=False, name='name')
    Params:
        y_test: list. The true labels
        y_pred: list. The model's labels
        class_names: list, the classes names'
        save: bool, Default = False. Saves the image as png
        name: string, default = 'name'. The file name if image is saved
    Returns:
        Returns confusion matrix plot
    """
    from pylab import rcParams
    rcParams['figure.figsize'] = 10, 10
    
    matrix = confusion_matrix(y_test, y_pred)
    plt.matshow(matrix, cmap=plt.cm.RdYlBu, aspect=1, alpha=0.6)
    
    # add color bar
    plt.colorbar()
    
    # add title and Axis Labels
    plt.title('Confusion Matrix', fontsize=20)
    plt.ylabel('Actual', fontsize=16)
    plt.xlabel('Predicted', fontsize=16)

    # add appropriate Axis Scales
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.grid(b=None)

    # add Labels to Each Cell
    thresh = matrix.max() / 2.  

    # iterate through the confusion matrix and append the labels
    for i, j in itertools.product(range(matrix.shape[0]), 
                                  range(matrix.shape[1])):
        plt.text(j, i, matrix[i, j], 
                 horizontalalignment="center",
                 color="black")

    # Add a Side Bar Legend Showing Colors
    
    plt.grid(b=None)
    if save:
        plt.grid(b=None)
        plt.tight_layout()
        plt.savefig(f'{model}_cm.png')
    return


def create_word_clouds(model, n=2, j=5, save=0, start=0, stop=None):
    """
    create_word_clouds(model, n, j, save, start, stop):
    Params:
        model: gensim LDA model object
        n: number of subplots in a column  (default=2)
        j: number of subplots in a row (default=5)
        save: save the figure (optional, default=0)
        start: from what number topic you wish to create the subplot (optional, default=0)
        stop: stop the subplot at a certain topic (optional, default=None)
        
    Returns:
        Word cloud image for every topic LDA created
    """
    # create color list
    colors_list = [color for name, color in mcolors.XKCD_COLORS.items()]
    
    # instantiate cloud
    cloud = WordCloud(background_color='white',
                      width=1028,
                      height=726,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: colors_list[start],
                      prefer_horizontal=1.0)
    
    # extract topics
    topics = model.show_topics(formatted=False)
    
    # create subplots 
    fig, axes = plt.subplots(n, j, figsize=(10,10), sharex=True, sharey=True)
    
    for ax in axes.flatten():
        fig.add_subplot(ax)
        topic_words = dict(topics[start][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(start+1), fontdict=dict(size=16))
        # hide axis
        plt.gca().axis('off')
        start += 1
        if start == stop:
            break
        
    plt.subplots_adjust(wspace=0, hspace=0)
    # hide axis
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    if save:
        plt.savefig(f'topics_cloud{start}.png')
    plt.show()


def print_topics(model, features, n):
    """
    print_topics(model, features, n):
    Params:
        model: sklearn LDA model object
        features: sklearn vectorizers.get_feature_names
        n: intenger - how many words to save/print for every topic
    Returns:
        Prints and saves a list of the 'n most important words for every topic
    """
    # make sure the features is in a numpy array to use .argsort
    if type(features) == list:
        features = np.array(features)
    
    # save the n most important words for each topic
    components = model.components_ 
    top_n = [features[component.argsort()][-n-1:] for component in components]
    
    # print the top words for every each topic
    for i in range(len(top_n)):
        print(f"Topic {i+1} most important words: {top_n[i]}")
    return top_n


def print_metrics(labels, predictions, print_score=False):
    """
    print_metrics(labels, predictions, print_score=False):
    Params:
        labels: list. the actual labels
        predictions: list, the model predictions
        print_score: bool, default = False.
    Returns:
        given actual labels and the predictions this function prints out the
        recall, accuracy, and f1 scores.
    """
    recall = round(recall_score(labels, predictions, average='micro')*100, 2)
    acc = round(accuracy_score(labels, predictions)*100, 2)
    f1 = round(f1_score(labels, predictions, average='micro')*100, 2)
    
    if print_score:
        print(f"Recall: {recall}")
        print(f"Accuracy: {acc}")
        print(f"F1 Score: {f1}")
    return 
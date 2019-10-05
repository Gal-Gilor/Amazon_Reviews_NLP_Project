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
import wordcloud
from wordcloud import WordCloud
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
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


def finalize_token(reviews, final_stopwords):
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
        clean = clean_review(review, final_stopwords)
        tokens = get_tokens(clean)
        lemmas = lem_words(tokens, lemmatizer)
        corpus.append(lemmas)
    return corpus


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    plot_confusion_matrix(cm, classes,normalize=False,
                          title='Confusion matrix',cmap=plt.cm.Blues)
    
    """
    #Add Normalization Option
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion')
    

def plot_AUC_ROC(y_score,fpr,tpr):
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    print('AUC: {}'.format(auc(fpr, tpr)))
    plt.figure(figsize=(10,8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_roc_curve(model, x_test, y_test):
    ''' This function accepts the model, testing set, testing labels, and outputs
        a Receiver Operating Characteristic curve plot'''
    # extract the target probability
    predict_proba = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, predict_proba)
    
    # plot the roc curve
    plt.figure(figsize=(8,5))
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC Curve')
   
    # plot a line through the origin of axis
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    
    # add graph labels
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve', fontsize=18)
    plt.legend(loc="lower right")
    return round(roc_auc_score(y_test, predict_proba), 2)

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
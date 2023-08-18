from nltk.corpus import brown, wordnet
from nltk.probability import FreqDist
from nltk import pos_tag
import nltk

def select_subset(subset_size):
    brown_words = [w.lower() for w in brown.words()]
    freq_dist = FreqDist(brown_words)

    common_words = {word for word, _ in freq_dist.most_common(subset_size)}
    common_words = {word for word in common_words if word.isalpha() and '-' not in word}

    wordnet_words = set(wordnet.words())
    selected_words = common_words.intersection(wordnet_words)

    # tagged_words = pos_tag(list(selected_words))
    # selected_words = {word for word, pos in tagged_words if pos in ['NN', 'VB', 'JJ', 'RB']}

    return selected_words

if __name__ == "__main__" :
    # nltk.download('averaged_perceptron_tagger')
    try:
        brown.ensure_loaded()
    except:
        nltk.download("brown")
    try:
        wordnet.ensure_loaded()
    except:
        nltk.download("wordnet")
    
    subset = select_subset(100)

    with open('corpora/corpus_subset.txt', 'w') as file:
        for word in subset:
            file.write(word + '\n')
from playgroundrl.client import *
from playgroundrl.actions import *
from playgroundrl.args import get_arguments
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import wordnet as word_corpus_loader
import nltk
import time

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

BOARD_SIZE = 25

try:
    word_corpus_loader.ensure_loaded()
except:
    nltk.download("wordnet")

nltk_corpus = set(word_corpus_loader.words())

model_path = 'models/glove-wiki-gigaword-200.gz'
glove_vectors = KeyedVectors.load_word2vec_format(model_path, binary=False)
# exit()
# def load_subset():
#     with open('corpora/corpus_subset_1000.txt', 'r') as file:
#         return {word.strip() for word in file}

# nltk_corpus = load_subset()

class TestCodenames(PlaygroundClient):
    def __init__(self, auth_file: str, render_gameplay: bool = False) -> None:
        super().__init__(
            GameType.CODENAMES,
            model_name='vector-one-word-3',
            auth_file=auth_file,
            render_gameplay=render_gameplay,
        )
        self.start_time = time.time()
        self.stopwatch = time.time()
        self.filtered_corpus = self.get_filtered_corpus()
    
    def get_filtered_corpus(self):
        filtered_words = set()
        for word in nltk_corpus:
            if word in glove_vectors and word.isalpha():
                filtered_words.add(word)
        return filtered_words

    def find_best_single_guess(self, state, clue_word, model, guesses):
        best_guess_index = 0
        best_similarity = -float('inf')

        clue_vector = self.get_word_vector(clue_word, model).reshape(1, -1)
        for i in range(BOARD_SIZE):
            # Skip if the word has already been guessed
            if state.guessed[i] != "UNKNOWN" or i in guesses:
                continue
            
            word = state.words[i]
            word_vector = self.get_word_vector(word, model)
            if word_vector is None:  # Skip if word_vector is None
                continue

            word_vector = word_vector.reshape(1, -1)
            similarity = cosine_similarity(clue_vector, word_vector)

            if similarity > best_similarity:
                best_guess_index = i
                best_similarity = similarity

        return best_guess_index
    
    def find_best_single_clue(self, target_word, model, board_words):
        best_clue = None
        best_similarity = -float('inf')
        target_vector = model.get_vector(target_word).reshape(1, -1)

        for word in self.filtered_corpus:
            if any(board_word in word or word in board_word for board_word in board_words):
                continue
            vector = model[word].reshape(1, -1)
            similarity = cosine_similarity(target_vector, vector)

            if similarity > best_similarity:
                best_clue = word
                best_similarity = similarity

        return best_clue, 1

    def get_target_word(self, state):
        for i in range(BOARD_SIZE):
            if (state.guessed[i] == "UNKNOWN") and (state.actual[i] == state.color):
                if state.words[i] in glove_vectors:
                    return state.words[i]
        return 'clue'

    def get_word_vector(self, word, model):
        words = word.split()
        vectors = [model[w] for w in words if w in model]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return None

    def callback(self, state: CodenamesState, reward):
        if state.player_moving_id not in self.player_ids:
            return None

        if state.role == "GIVER":
            target_word = self.get_target_word(state)
            print(f'Target word: {target_word}')
            word, count = self.find_best_single_clue(target_word, glove_vectors, state.words) # Find the best clue based on word embeddings
            print('Best clue: ', word)
            return CodenamesSpymasterAction(word=word, count=count)
        elif state.role == "GUESSER":
            clue_word = state.clue
            guesses = []
            # for _ in range(state.count):
            for _ in range(1):
                print(f'Guessing on {clue_word}')
                guess_idx = self.find_best_single_guess(state, clue_word, glove_vectors, guesses)
                guesses.append(guess_idx)
                print('I think it is', state.words[guess_idx])
            print('stopwatch: ', time.time() - self.stopwatch)
            self.stopwatch = time.time()
            print()
            return CodenamesGuesserAction(guesses)


    def gameover_callback(self) -> None:
        print('total time: ', time.time() - self.start_time)


if __name__ == "__main__":
    init_args, run_args = get_arguments()
    t = TestCodenames(**init_args)
    for _ in range(100):
        t.run(**run_args)
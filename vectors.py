from playgroundrl.client import *
from playgroundrl.actions import *
from util import parse_arguments
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import wordnet as word_corpus_loader
import nltk

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

BOARD_SIZE = 25

try:
    word_corpus_loader.ensure_loaded()
except:
    nltk.download("wordnet")

nltk_corpus = set(word_corpus_loader.words())

class TestCodenames(PlaygroundClient):
    def __init__(self, auth_file: str, render: bool = False) -> None:
        super().__init__(
            GameType.CODENAMES,
            model_name='vector-naive',
            auth_file=auth_file,
            render_gameplay=render,
        )

    def find_best_clue(self, state: CodenamesState, model):
        team_words = [model[word] for word in state.team_words]
        opponent_words = [model[word] for word in state.opponent_words]
        bomb_word = model[state.bomb_word]
        
        avg_vector = np.mean(team_words, axis=0).reshape(1, -1)
        
        best_clue = None
        best_similarity = -float('inf')
        
        for word in nltk_corpus:
            vector = model[word].reshape(1, -1)
            similarity = cosine_similarity(avg_vector, vector)
            dissimilarity = min(cosine_similarity(vector, w.reshape(1, -1)) for w in opponent_words + [bomb_word])
            
            if similarity > best_similarity and dissimilarity < threshold: # threshold???
                best_clue = word
                best_similarity = similarity
        
        count = len([w for w in team_words if cosine_similarity(avg_vector, w.reshape(1, -1)) > some_threshold]) # threshold???
        return best_clue, count

    def callback(self, state: CodenamesState, reward):
        if state.player_moving_id not in self.player_ids:
            return None

        if state.role == "GIVER":
            model_path = 'models/glove-twitter-25.gz'
            glove_vectors = KeyedVectors.load_word2vec_format(model_path, binary=False)
            word, count = self.find_best_clue(state, glove_vectors) # Find the best clue based on word embeddings
            return CodenamesSpymasterAction(word=word, count=count)

    def gameover_callback(self) -> None:
        pass


if __name__ == "__main__":
    # args = parse_arguments("codenames")
    args = parse_arguments()
    t = TestCodenames(args.authfile, args.render)
    t.run(
        # pool=Pool(args.pool),
        pool=Pool.MODEL_ONLY,
        num_games=args.num_games,
        self_training=args.self_training,
        maximum_messages=500000,
        # used to set up 2-player game (rather than default 4)
        game_parameters={"num_players": 2},
    )

# ['europe', 'giant', 'cover', 'theater', 'ship', 'parachute', 'temple', 'gas', 'ghost']
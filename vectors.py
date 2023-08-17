from playgroundrl.client import *
from playgroundrl.actions import *

from gensim.models import Word2Vec

class TestCodenames(PlaygroundClient):
    def __init__(self) -> None:
        super().__init__(
            GameType.CODENAMES,
            model_name='vector-codenames',
            auth_file=auth_file
            render_gameplay=render,
        )

    def callback(self, state: CodenamesState, reward):
        if state.player_moving_id not in self.player_ids:
            return None

        if state.role == "GIVER":
            model = Word2Vec.load('path/to/model') # Load a pre-trained Word2Vec model
            word, count = self.find_best_clue(state, model) # Find the best clue based on word embeddings
            return CodenamesSpymasterAction(word=word, count=count)

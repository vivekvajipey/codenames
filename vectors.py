from playgroundrl.client import *
from playgroundrl.actions import *
from util import parse_arguments
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
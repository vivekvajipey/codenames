from playgroundrl.client import *
from playgroundrl.actions import *
from util import parse_arguments

BOARD_SIZE = 25

class TestCodenames(PlaygroundClient):
    def __init__(self, auth_file: str, render: bool = False):
        super().__init__(
            GameType.CODENAMES,
            model_name="tutorial-codenames",
            auth_file=auth_file,
            render_gameplay=render,
        )

    def get_open_squares(self, state: CodenamesState):
        open_squares = []
        for i in range(BOARD_SIZE):
            if state.guessed[i] == "UNKNOWN":
                open_squares.append(i)
        return open_squares

    def callback(self, state: CodenamesState, reward):
        if state.player_moving_id not in self.player_ids:
            return None

        if state.role == "GIVER":
            return CodenamesSpymasterAction(
                word = "clue", 
                count = 1 
            )
        elif state.role == "GUESSER":
            return CodenamesGuesserAction(
                guesses= self.get_open_squares(state)[:2],
            )

    def gameover_callback(self):
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
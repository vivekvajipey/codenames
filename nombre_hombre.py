from playgroundrl.client import *
from playgroundrl.actions import *
from playgroundrl.args import get_arguments

BOARD_SIZE = 25

class TestCodenames(PlaygroundClient):
    def __init__(self, auth_file: str, render: bool = False):
        super().__init__(
            GameType.CODENAMES,
            model_name="codenombre-hombre",
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
    init_args, run_args = get_arguments()
    t = TestCodenames(**init_args)
    t.run(**run_args)
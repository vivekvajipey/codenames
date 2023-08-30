from playgroundrl.client import *
from playgroundrl.actions import *
from playgroundrl.args import get_arguments
import openai
import os

BOARD_SIZE = 25
openai.api_key = os.environ.get('GPT3_API_KEY')


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

    def generate_clue(self, state: CodenamesState):
        """Gets a clue from the GPT 3.5 API."""
        words = self.get_open_squares(state)

    def generate_guess(self, state: CodenamesState):
        pass
    
    def callback(self, state: CodenamesState, reward):
        if state.player_moving_id not in self.player_ids:
            return None

        if state.role == "GIVER":
            gpt_word, gpt_count = self.generate_clue(state)
            return CodenamesSpymasterAction(word=gpt_word, count=gpt_count)
        elif state.role == "GUESSER":
            gpt_guesses = self.generate_guess(state)
            return CodenamesGuesserAction(guesses=gpt_guesses)

    def gameover_callback(self):
        pass

if __name__ == "__main__":
    # args = parse_arguments("codenames")
    init_args, run_args = get_arguments()
    t = TestCodenames(**init_args)
    t.run(**run_args)
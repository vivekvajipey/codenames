import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="TestCodenames script")
    parser.add_argument("--authfile", help="Path to the authentication file", required=True)
    parser.add_argument("--render", help="Render gameplay", action="store_true")
    parser.add_argument("--pool", help="Pool type")
    parser.add_argument("--num_games", help="Number of games to play", type=int)
    parser.add_argument("--self_training", help="Enable self-training", action="store_true")
    return parser.parse_args()
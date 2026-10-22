import sys

try:
    from .agent_loop import run
except ImportError:
    from agent_loop import run


if __name__ == "__main__":
    default = "What is (12+8)-3, and how many words are in 'hello tiny agent'?"
    print(run(" ".join(sys.argv[1:]) or default))

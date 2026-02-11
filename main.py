import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <assignment_number> [args...]")
        print("  e.g. python main.py 1")
        sys.exit(1)

    assignment = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # forward remaining args

    if assignment == "1":
        from assignment_1 import main as run
        run()
    else:
        print(f"Unknown assignment: {assignment}")
        sys.exit(1)


if __name__ == "__main__":
    main()

import os

from service import greet


def main() -> None:
    name = os.getenv("APP_GREETING", "Harness")
    print(greet(name))


if __name__ == "__main__":
    main()


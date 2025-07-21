import importlib.metadata

version = importlib.metadata.version("oamc")

text = rf"""
  ____  ___   __  ________
 / __ \/ _ | /  |/  / ___/
/ /_/ / __ |/ /|_/ / /__
\____/_/ |_/_/  /_/\___/

Optimal Additive Manufacturing of Composites
Version {version}

Copyright (c) 2025 Nicolas Ebeling
MIT License
"""


def banner() -> None:
    print(text)

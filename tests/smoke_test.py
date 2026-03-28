import importlib


def test_import_package() -> None:
    importlib.import_module("oamc")

if __name__ == "__main__":
    test_import_package()

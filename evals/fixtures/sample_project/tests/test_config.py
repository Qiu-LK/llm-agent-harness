from src.config import APP_NAME


def test_config_is_a_string() -> None:
    assert isinstance(APP_NAME, str)

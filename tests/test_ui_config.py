from manga_translator.ui import build_app


def test_setup_status_is_collapsed_by_default():
    app = build_app()
    config = app.get_config_file()
    accordions = [
        component
        for component in config["components"]
        if component.get("type") == "accordion"
    ]

    setup = next(component for component in accordions if component["props"].get("label") == "Setup Status")
    assert setup["props"].get("open") is False

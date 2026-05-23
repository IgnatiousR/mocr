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


def test_preview_images_are_inside_tab_items():
    app = build_app()
    config = app.get_config_file()
    tab_items = [
        component
        for component in config["components"]
        if component.get("type") == "tabitem"
    ]

    assert [item["props"].get("label") for item in tab_items] == [
        "Original",
        "Detected boxes",
        "Cleaned",
        "Final composite",
    ]


def test_internal_callbacks_are_hidden_from_api_docs():
    app = build_app()
    config = app.get_config_file()

    assert all(dependency.get("show_api") is False for dependency in config["dependencies"])

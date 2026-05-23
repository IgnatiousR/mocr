from manga_translator.ui import build_app


def test_ocr_setup_sidebar_section_is_collapsed_by_default():
    app = build_app()
    config = app.get_config_file()
    accordions = [
        component
        for component in config["components"]
        if component.get("type") == "accordion"
    ]

    setup = next(component for component in accordions if component["props"].get("label") == "OCR and setup")
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


def test_settings_sidebar_and_main_workspace_are_present():
    app = build_app()
    config = app.get_config_file()
    class_sets = [
        component.get("props", {}).get("elem_classes") or []
        for component in config["components"]
    ]

    assert any("settings-sidebar" in classes for classes in class_sets)
    assert any("main-workspace" in classes for classes in class_sets)


def test_translation_backend_uses_env_default(monkeypatch):
    monkeypatch.setenv("TRANSLATION_BACKEND", "sugoi")
    app = build_app()
    config = app.get_config_file()
    backend = next(
        component
        for component in config["components"]
        if component.get("type") == "dropdown"
        and component.get("props", {}).get("label") == "translation backend"
    )

    assert backend["props"].get("value") == "sugoi"

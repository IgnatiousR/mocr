from manga_translator import ui
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


def test_app_uses_stone_theme():
    app = build_app()

    assert app.theme.name == "theme-repo/STONE_Theme"


def test_ram_usage_sidebar_section_is_visible_by_default():
    app = build_app()
    config = app.get_config_file()
    accordions = [
        component
        for component in config["components"]
        if component.get("type") == "accordion"
    ]

    ram = next(component for component in accordions if component["props"].get("label") == "RAM")
    assert ram["props"].get("open") is True


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
    assert all(dependency.get("api_name") is False for dependency in config["dependencies"])
    assert app.get_api_info() == {"named_endpoints": {}, "unnamed_endpoints": {}}


def test_settings_sidebar_and_main_workspace_are_present():
    app = build_app()
    config = app.get_config_file()
    class_sets = [
        component.get("props", {}).get("elem_classes") or []
        for component in config["components"]
    ]

    assert any("settings-sidebar" in classes for classes in class_sets)
    assert any("main-workspace" in classes for classes in class_sets)


def test_translation_model_preset_uses_env_default(monkeypatch):
    monkeypatch.setenv("TRANSLATION_BACKEND", "sugoi")
    app = build_app()
    config = app.get_config_file()
    preset = next(
        component
        for component in config["components"]
        if component.get("type") == "dropdown"
        and component.get("props", {}).get("label") == "Translation model"
    )

    assert preset["props"].get("value") == "sugoi-v4"


def test_translation_model_preset_includes_expected_options():
    app = build_app()
    config = app.get_config_file()
    preset = next(
        component
        for component in config["components"]
        if component.get("type") == "dropdown"
        and component.get("props", {}).get("label") == "Translation model"
    )

    choices = preset["props"].get("choices")
    assert ("Gemma 2 2B GGUF (llama.cpp)", "gemma-gguf") in choices
    assert ("Sugoi V4 CTranslate2", "sugoi-v4") in choices
    assert ("Fugu-MT Transformers", "fugumt") in choices


def test_translation_backend_is_hidden_internal_field():
    app = build_app()
    config = app.get_config_file()
    backend = next(
        component
        for component in config["components"]
        if component.get("type") == "textbox"
        and component.get("props", {}).get("label") == "translation backend"
    )

    assert backend["props"].get("visible") is False


def test_translation_preset_change_returns_backend_path_and_status(monkeypatch):
    monkeypatch.delenv("TRANSLATION_MODEL_PATH", raising=False)
    backend, model_path, setup_status, status = ui.select_translation_preset(
        "gemma-gguf",
        "",
        "",
        "opencv-telea",
        "",
        "outputs",
    )

    assert backend == "llama"
    assert model_path == "models/translation/gemma-2-2b-jpn-it-translate-Q4_K_M.gguf"
    assert "Setup Status" in setup_status
    assert "Gemma 2 2B GGUF" in status


def test_review_dataframe_uses_array_rows():
    app = build_app()
    config = app.get_config_file()
    review = next(
        component
        for component in config["components"]
        if component.get("type") == "dataframe"
        and component.get("props", {}).get("label") == "Review and edit regions"
    )

    assert review["props"].get("type") == "array"


def test_inpainter_controls_are_present():
    app = build_app()
    config = app.get_config_file()
    inpainter = next(
        component
        for component in config["components"]
        if component.get("type") == "dropdown"
        and component.get("props", {}).get("label") == "Inpainter"
    )
    inpaint_path = next(
        component
        for component in config["components"]
        if component.get("type") == "textbox"
        and component.get("props", {}).get("label") == "Inpaint model path"
    )
    download = next(
        component
        for component in config["components"]
        if component.get("type") == "button"
        and component.get("props", {}).get("value") == "Download inpainter model"
    )

    assert inpainter["props"].get("value") == "opencv-telea"
    assert inpaint_path["props"].get("value") == ""
    assert download is not None


def test_render_gap_and_visibility_controls_are_present():
    app = build_app()
    config = app.get_config_file()
    labels = {
        component.get("props", {}).get("label")
        for component in config["components"]
    }

    assert "text box gap" in labels
    assert "line gap" in labels
    assert "show all translated text" in labels


def test_batch_results_gallery_is_present():
    app = build_app()
    config = app.get_config_file()
    gallery = next(
        component
        for component in config["components"]
        if component.get("type") == "gallery"
        and component.get("props", {}).get("label") == "Batch results"
    )

    assert gallery is not None


def test_batch_callback_output_count_matches_return_tuple():
    app = build_app()
    config = app.get_config_file()
    dependencies = config["dependencies"]
    batch_dependency = next(
        dependency
        for dependency in dependencies
        if len(dependency.get("outputs") or []) == 10
    )

    assert len(batch_dependency["inputs"]) == 21
    assert len(batch_dependency["outputs"]) == 10

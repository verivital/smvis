"""Tests for the Dash application."""


class TestAppCreation:
    def test_create_app(self):
        from smvis.app import create_app
        app = create_app()
        assert app is not None

    def test_app_has_layout(self):
        from smvis.app import create_app
        app = create_app()
        assert app.layout is not None

    def test_model_files_found(self):
        from smvis.app import _find_model_files
        files = _find_model_files()
        assert len(files) >= 4
        assert "counter.smv" in files
        assert "mutex.smv" in files

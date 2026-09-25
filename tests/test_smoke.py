def test_all_modules_import_and_app_builds():
    import importlib

    modules = [
        "app.api",
        "app.calibration",
        "app.enums",
        "app.errors",
        "app.models",
        "app.numerics",
        "app.pricing.barrier",
        "app.pricing.black_scholes",
        "app.service",
        "app.validation",
    ]
    imported = [importlib.import_module(name) for name in modules]
    assert len(imported) == len(modules)

    from app.api import create_app

    assert create_app() is not None

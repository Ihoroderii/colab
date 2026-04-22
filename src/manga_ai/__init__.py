# Manhwa AI package
__all__ = ["render_panel_spec"]


def render_panel_spec(config, panel_spec: dict, ip_adapter_images=None):
    """Render a panel from a PanelSpec dict. See runners.main for full docs."""
    from .runners.main import render_panel_spec as _impl
    return _impl(config, panel_spec, ip_adapter_images)

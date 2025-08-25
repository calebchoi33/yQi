"""Legacy UI components - DEPRECATED. Use ui.components instead."""

# This file is deprecated. All UI components have been moved to ui/components.py
# Import from there instead:
# from ui.components import UIComponents

import warnings
warnings.warn(
    "ui_components.py is deprecated. Use 'from ui.components import UIComponents' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Legacy imports for backward compatibility
from ui.components import (
    load_api_key,
    display_responses, 
    display_benchmark_summary,
    render_download_button
)

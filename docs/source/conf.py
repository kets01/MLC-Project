# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MLC Lab'
copyright = '2026, "Ketsia Kemkuini" "Mariza Yamdjeu"'
author = '"Ketsia Kemkuini" "Mariza Yamdjeu"'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/kets01/MLC-Project",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "light_css_variables": {
        "color-brand-primary": "#0f6f6a",
        "color-brand-content": "#0f6f6a",
        "font-stack--monospace": "'SF Mono', 'Menlo', 'Consolas', monospace",
        # Own variable (not a furo built-in) so custom.css can soften heading
        # color without hard-coding a hex that would look wrong in dark mode.
        "color-heading-text": "#2b2f33",
    },
    "dark_css_variables": {
        # Furo's default dark mode is near-black with a bright cyan accent —
        # too harsh a contrast. Warm the background up and mute the accent.
        "color-brand-primary": "#6fbfb5",
        "color-brand-content": "#6fbfb5",
        "color-background-primary": "#1c1e21",
        "color-background-secondary": "#212428",
        "color-background-hover": "#26292d",
        "color-background-border": "#33373c",
        "color-foreground-primary": "#d6d9dc",
        "color-foreground-secondary": "#9a9ea3",
        "color-heading-text": "#e4e6e8",
    },
}

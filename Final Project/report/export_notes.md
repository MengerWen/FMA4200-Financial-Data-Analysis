# Export Notes

## Status

Automatic PDF export is not supported in the current environment.

## Generated Files

- `report/final_report.md`
- `report/final_report.ipynb`
- `report/final_report.html`
- `report/final_appendices.md`

## Why `final_report.pdf` Was Not Generated Automatically

The environment check for this report build found the following PDF-related tools on PATH:

- `pandoc`: None
- `xelatex`: None
- `pdflatex`: None
- `wkhtmltopdf`: None
- `chromium`: None
- `chrome`: None
- `msedge`: None

In this environment, the report build can produce clean Markdown, notebook, and HTML sources, but it does not have the non-Python PDF toolchain needed for a reliable automated PDF export.

## Final Export Step

1. Open `report/final_report.html` in a browser.
2. Use the browser's **Print** command.
3. Choose **Save as PDF** and save the file as `report/final_report.pdf`.

The HTML file is already formatted as the print-ready source for this final manual step.

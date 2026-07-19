# Final Report LaTeX Source

This directory is an IEEE two-column conference-paper scaffold. It intentionally contains no AI-generated report prose. Human authors must replace only the marked `% [作者手写区]` regions and must preserve the page-budget comments and evidence constraints.

Compile from this directory after Tectonic has its packages available:

```bash
cd latex_source
tectonic main.tex
```

Directory structure:

- `main.tex`: IEEEtran document wiring, title, bibliography, and figure fallback macro.
- `sections/`: one file per required section. Every section after the abstract starts with `\clearpage`; the platform table's "Hengyuan Cloud" is the ASCII rendering of 恒源云.
- `figures/`: final `fig1.pdf` through `fig8.pdf` and the Appendix A screenshot PDF. Until those files exist, `main.tex` renders labeled placeholder boxes.
- `refs.bib`: IEEE/BibTeX records derived from the three approved CSV evidence files. Missing bibliographic fields are explicit TODO comments.
- `../scripts/plot_final_report_figures.py`: shared Matplotlib style, figure registry, and the system-architecture generators for Fig.1--Fig.2. Fig.3--Fig.8 intentionally refuse to generate until their measured inputs are supplied.

Before submission, the human authors must compile and adjust only human-authored text/figure sizing until the binary page budgets are met, replace the Appendix A placeholder with a GitHub screenshot, wire every claim to a figure/table/citation, remove `\nocite{*}`, and verify that no bullet list exists outside Introduction. Figure axes, ticks, and legends must remain at least 10 pt.

#!/usr/bin/env bash

mkdir out

mv out/* thesis/
cd thesis/

pdflatex title.tex
mv *.aux *.bbl *.blg *.brf *.ilg *.loa *.lof *.log *.lot *.nlo *.nls *.out *.pdf *.toc ../out/

pdflatex main.tex
bibtex main
pdflatex main.tex
bitex main
makeindex main.nlo -s nomencl.ist -o main.nls
pdflatex main.tex
pdflatex main.tex
mv *.aux *.bbl *.blg *.brf *.ilg *.loa *.lof *.log *.lot *.nlo *.nls *.out *.pdf *.toc ../out/
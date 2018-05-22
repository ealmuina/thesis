#!/usr/bin/env bash

mkdir auxil out

mv auxil/* thesis/
cd thesis/

pdflatex title.tex
mv *.pdf ../out/
mv *.aux *.bbl *.blg *.brf *.ilg *.loa *.lof *.log *.lot *.nlo *.nls *.out *.toc ../auxil/

pdflatex main.tex
bibtex main
pdflatex main.tex
bitex main
makeindex main.nlo -s nomencl.ist -o main.nls
pdflatex main.tex
pdflatex main.tex
mv *.pdf ../out/
mv *.aux *.bbl *.blg *.brf *.ilg *.loa *.lof *.log *.lot *.nlo *.nls *.out *.toc ../auxil/
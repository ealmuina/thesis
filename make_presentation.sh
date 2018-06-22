#!/usr/bin/env bash

mkdir out

cd thesis/presentation/
pdflatex presentation.tex
pdflatex presentation.tex

mv *.pdf ../../out/
rm *.aux *.log *.nav *.snm *.out *.toc *.vrb



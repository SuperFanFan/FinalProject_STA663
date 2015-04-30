STA663_Final_Report_Li_Fan.pdf: STA663_Final_Report_Li_Fan.tex
	pdflatex STA663_Final_Report_Li_Fan
	pdflatex STA663_Final_Report_Li_Fan
	pdflatex STA663_Final_Report_Li_Fan

.PHONY: all clean allclean test

all: STA663_Final_Report_Li_Fan.pdf 

clean:
	rm -rf *aux *log *pytxcode UnitTest/__pycache__ UnitTest/*pyc

allclean:
	make clean
	rm -f *pdf

test:
	py.test ~/FinalProject_STA663/UnitTest/.
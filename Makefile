sources = $(shell find -name "*.f90")
deps = sources.bib

.PRECIOUS: *.dat

all:
	mkdir -p data
	#make build
	make $(deps)
	make report.pdf

#build: $(sources)
#	mkdir -p build && cd build && FC=caf cmake -DCMAKE_BUILD_TYPE=Release .. && make

%.pdf: %.tex $(deps)
	latexmk -pdflua -time -g -shell-escape $*

%.pdf: %.asy
	asy -maxtile "(400,400)" -o $@ $<

clean:
	latexmk -c
	rm -rf *.run.xml *.bbl build
	find -name "*.mod" -delete

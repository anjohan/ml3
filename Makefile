sources = $(shell find -name "*.f90")
deps = sources.bib $(foreach dx,2.0E-01 1.0E-02,data/euler_$(dx).dat)

.PRECIOUS: *.dat

all:
	mkdir -p data
	make build
	make $(deps)
	make report.pdf

build: $(sources)
	mkdir -p build && cd build && FC=caf cmake -DCMAKE_BUILD_TYPE=Release .. && make

%.pdf: %.tex $(deps)
	latexmk -pdflua -time -g -shell-escape $*

%.pdf: %.asy
	asy -maxtile "(400,400)" -o $@ $<

data/euler_%.dat: build/euler
	./$< <<< $*

build/%: build

clean:
	latexmk -c
	rm -rf *.run.xml *.bbl build
	find -name "*.mod" -delete

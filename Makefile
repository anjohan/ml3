sources = $(shell find -name "*.f90")
deps = sources.bib $(foreach dx,2.0E-01 1.0E-02,data/euler_$(dx).dat)\
	   $(foreach u,1 2,data/nn_cost_table_small_$(u).dat)\
	   data/nn_u_100_1.dat data/nn_u_100_2.dat

.PRECIOUS: *.dat

all:
	mkdir -p data
	make build
	make $(deps)
	make report.pdf

build: $(sources)
	mkdir -p build && cd build && FC=caf cmake -DCMAKE_BUILD_TYPE=Release .. && make

%.pdf: %.tex $(deps)
	latexrun --latex-cmd lualatex --bibtex-cmd biber $*

%.pdf: %.asy
	asy -maxtile "(400,400)" -o $@ $<

data/euler_%.dat: build/euler
	./$< <<< $*

data/nn_cost_table_small_%.dat: programs/nn_params_analysis.py data/nn_costs_%.dat
	python $< $*

data/nn_costs_%.dat: Makefile.costs programs/nn_params.py
	make -f $< $@

data/nn_u_100_%.dat: programs/nn_simple.py
	python $< $*

build/%: build

clean:
	latexmk -c
	rm -rf *.run.xml *.bbl build
	find -name "*.mod" -delete

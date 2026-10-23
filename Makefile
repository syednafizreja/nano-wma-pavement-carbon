.PHONY: all selftest test reproduce abstract
all: test reproduce
selftest:
	python run_all.py --selftest
test:
	pytest -q
reproduce:
	python run_all.py --outdir results
	python docs/make_graphical_abstract.py

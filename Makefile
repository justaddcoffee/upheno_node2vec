all: data

data: data/hp-to-mp-bestmatches.tsv data/hp.nt data/mp.nt

data/hp-to-mp-bestmatches.tsv:
	mkdir data
	wget https://github.com/obophenotype/upheno/blob/master/mappings/hp-to-mp-bestmatches.tsv?raw=true -O data/hp-to-mp-bestmatches.tsv

data/hp.owl:
	wget http://purl.obolibrary.org/obo/hp.owl -O data/hp.owl

data/mp.owl:
	wget http://www.informatics.jax.org/downloads/reports/mp.owl -O data/mp.owl

data/hp.nt: data/hp.owl
	rapper -o ntriples data/hp.owl > data/hp.nt

data/mp.nt: data/mp.owl
	rapper -o ntriples data/mp.owl > data/mp.nt

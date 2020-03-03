all: data

data: data/hp-to-mp-bestmatches.edges data/hp.edges data/mp.edges \
data/upheno_root.edges data/go.edges data/mpath.edges data/nbo.edges data/uberon.edges \
data/all.edges data/curie_map.yaml

data/hp-to-mp-bestmatches.edges:
	mkdir -p data
	wget 'https://github.com/obophenotype/upheno/blob/master/mappings/hp-to-mp-bestmatches.tsv?raw=true' -O data/hp-to-mp-bestmatches.tsv
	cut -f1,3,5 data/hp-to-mp-bestmatches.tsv > data/hp-to-mp-bestmatches.edges

data/hp.edges:
	wget http://purl.obolibrary.org/obo/hp.owl -O data/hp.owl
	rapper -o ntriples data/hp.owl | cut -f1,3 -d " " > data/hp.edges

data/mp.edges:
	wget http://www.informatics.jax.org/downloads/reports/mp.owl -O data/mp.owl
	rapper -o ntriples data/mp.owl | cut -f1,3 -d " " > data/mp.edges

data/upheno_root.edges:
	wget https://raw.githubusercontent.com/obophenotype/upheno/master/upheno_root_alignments.owl -O data/upheno_root.owl
	grep SubClassOf data/upheno_root.owl | perl -p -e 's/SubClassOf\(//' | perl -p -e 's/\)//' >> data/upheno_root.edges

data/go.edges:
	wget http://purl.obolibrary.org/obo/upheno/imports/go_phenotype.owl -O data/go.owl
	rapper -o ntriples data/go.owl | cut -f1,3 -d " " > data/go.edges

data/mpath.edges:
	wget http://purl.obolibrary.org/obo/upheno/imports/mpath_phenotype.owl -O data/mpath.owl
	rapper -o ntriples data/mpath.owl | cut -f1,3 -d " " > data/mpath.edges

data/nbo.edges:
	wget http://purl.obolibrary.org/obo/upheno/imports/nbo_phenotype.owl -O data/nbo.owl
	rapper -o ntriples data/nbo.owl | cut -f1,3 -d " " > data/nbo.edges

data/uberon.edges:
	wget http://purl.obolibrary.org/obo/upheno/imports/uberon_phenotype.owl -O data/uberon.owl
	rapper -o ntriples data/uberon.owl | cut -f1,3 -d " " > data/uberon.edges

data/all.edges: data/hp.edges data/mp.edges data/upheno_root.edges data/go.edges data/mpath.edges data/nbo.edges data/uberon.edges
	cat data/hp.edges data/mp.edges data/upheno_root.edges data/go.edges data/mpath.edges data/nbo.edges data/uberon.edges > data/all.edges

data/curie_map.yaml:
	wget https://raw.githubusercontent.com/monarch-initiative/dipper/master/dipper/curie_map.yaml -O data/curie_map.yaml

all: data

data: mkdir data/hp-to-mp-bestmatches.edges data/upheno.edges data/curie_map.yaml

mkdir:
	mkdir -p data

data/hp-to-mp-bestmatches.edges:
	cut -f1,2 -d"," attic/upheno_mapping_mp_hp.csv | perl -p -e 's/,/\t/' | tail -n +2 > data/hp-to-mp-bestmatches.edges

data/upheno.edges:
	wget https://ci.monarchinitiative.org/view/pipelines/job/upheno2/81/artifact/src/curation/upheno-release/mp-hp/upheno_mp-hp_with_relations.owl -O data/upheno_mp-hp_with_relations.owl
	rapper -o ntriples data/upheno_mp-hp_with_relations.owl | cut -f1,3 -d " " > data/upheno.edges

data/curie_map.yaml:
	wget https://raw.githubusercontent.com/monarch-initiative/dipper/master/dipper/curie_map.yaml -O data/curie_map.yaml

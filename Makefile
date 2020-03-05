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

testdata: mkdir data
	head -3000 data/upheno_mp-hp_with_relations.owl > data/upheno_mp-hp_with_relations_SNIPPET.owl
	rapper -o ntriples data/upheno_mp-hp_with_relations_SNIPPET.owl | cut -f1,3 -d " " > data/upheno_SNIPPET.edges

testrun: testdata
	echo "trivial test run that runs link prediction on the first few thousand edges from the upheno owl file"
	python runLinkPrediction_pheno_match.py --upheno_graph data/upheno_SNIPPET.edges --equivalent_phenotypes data/hp-to-mp-bestmatches.edges --weight_multiplier 10

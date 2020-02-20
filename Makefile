all: data

data:
	# sneak into Nico's Jenkins pipeline
	scp -Cpr m5:/var/lib/jenkins/jobs/upheno2/builds/60/archive/src/curation/upheno-release/all/* data/
	scp -Cpr m5:/var/lib/jenkins/jobs/upheno2/builds/60/archive/src/curation/upheno-release/mp-hp/* data/
	wget https://github.com/obophenotype/upheno/blob/master/mappings/hp-to-mp-bestmatches.tsv?raw=true -O data/hp-to-mp-bestmatches.tsv


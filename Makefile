all: data

data:
	# sneak into Nico's Jenkins pipeline
	scp -Cpr m5:/var/lib/jenkins/jobs/upheno2/builds/60/archive/src/curation/upheno-release/all/* data/
	scp -Cpr m5:/var/lib/jenkins/jobs/upheno2/builds/60/archive/src/curation/upheno-release/mp-hp/* data/

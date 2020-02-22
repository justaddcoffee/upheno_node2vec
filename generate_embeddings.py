import wget
import os
import logging
from owlready2 import get_ontology
from urllib.parse import urlparse
import n2v

logging.basicConfig(level=logging.DEBUG)

hp_owl_iri = 'http://purl.obolibrary.org/obo/hp.owl'
mp_owl_iri = 'http://www.informatics.jax.org/downloads/reports/mp.owl'
hp_to_mp_bestmatch_url = 'https://github.com/obophenotype/upheno/blob/master/mappings/hp-to-mp-bestmatches.tsv?raw=true'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(BASE_DIR, 'data')


def owl2edge(owl_iri: str, data_dir: str) -> bool:
    a = urlparse(owl_iri)
    local_file = os.path.join(data_dir, os.path.basename(a.path))
    if not os.path.exists(local_file):
        logging.info("downloading %s", owl_iri)
        wget.download(owl_iri, local_file)

    edge_file = local_file + ".edge"
    if not os.path.exists(edge_file):
        with open(local_file, 'r') as owl_file, open(edge_file, 'w') as edge:
            onto = get_ontology(local_file).load()
            for s, p, o, d in onto.graph._iter_triples():
                try:
                    line = "\t".join([onto._unabbreviate(s), onto._unabbreviate(o), str(1)])
                except KeyError as e:
                    logging.ERROR("keyerror: %s", e)
                edge.writelines(line)
    return True


for owl_iri in [hp_owl_iri, mp_owl_iri]:
    owl2edge(owl_iri, data_dir)





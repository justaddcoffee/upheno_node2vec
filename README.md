Download upheno into graph and make MP to HP phenotype-phenotype link predictions

```
git clone https://github.com/justaddcoffee/upheno_node2vec.git
cd upheno_node2vec
port install rapper # or brew install rapper or maybe apt-get install rapper
make
pip install -r requirements.txt
pip install -e git+git://github.com/justaddcoffee/N2V.git@phenotype_matching#egg=n2v
python runLinkPrediction_pheno_match.py --upheno_graph data/upheno.edges --equivalent_phenotypes data/hp-to-mp-bestmatches.edges --weight_multiplier 10
```

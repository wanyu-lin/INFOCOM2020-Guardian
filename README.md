# This repo covers the implementation for our paper Guardian.
Wanyu Lin, Zhaolin Gao and Baochun Li, "[Guardian: Evaluating Trust in Online Social Networks with Graph Convolutional Networks](https://wanyu-lin.github.io/assets/publications/wlin-infocom20.pdf)," in the Proceedings of IEEE INFOCOM, Toronto, Canada, July 6-9, 2020.

## Download
```sh  
git clone 
cd wanyu-infocom20-code  
```
This directory contains code necessary to run Guardian algorithm. It is especially useful for massive trust assessment in online social networks. 


## Setup environment
The codebase is implemented in Python 3.6.7. You can install all the required packages using the following command:

	$ pip install -r requirements.txt


## Run the code
```sh
cd /guardian_code/
python main.py [--argment arg_val]
```
## Cite
If you make advantage in your research, please cite the following in your manuscript:
```
@INPROCEEDINGS{9155370,
  author={Lin, Wanyu and Gao, Zhaolin and Li, Baochun},
  booktitle={in the Proceedings of IEEE INFOCOM}, 
  title={Guardian: Evaluating Trust in Online Social Networks with Graph Convolutional Networks}, 
  year={2020},
  doi={10.1109/INFOCOM41043.2020.9155370}}
```

## Datasets

This directory (/data) contains the related files of the trust-based social networks -- Advogato and PGP. 

The first data set is Advogato. It is a trust-based social network for open source developers. To allow users to certify each other, the network provides 4 levels of trust assertions, i.e., ‘Observer’, ‘Apprentice’, ‘Journeyer’, and ‘Master’.

The second data set is PGP (short for Pretty Good Privacy). PGP adopts the concept of ‘web of trust’ to establish a decentralized model for data encryption and decryption.
	
	advogato-graph-2014-03-28CLEAN-.txt -- metadata about advogato (http://www.trustlet.org/wiki/Advogato_dataset)
    PGP_nx.edgelist -- metadata about PGP (http://networkrepository.com/tech-pgp.php)
    

 The fist column of nodes are trusters, second column of nodes are trustees, and what follows are associated trust levels (‘Observer’,‘Apprentice’, ‘Journeyer’, and ‘Master’)

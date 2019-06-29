# King-Recommender
This thesis purpose to improve movies recommender systems with visual information. Visual information from posters is extracted as features, from features are created clusters mapped to the total number of genres from movies dataset.

The implementation of system can be represented through three steps: 
1. Collecting the posters for the movies from dataset;
2. Feature extraction and clusters creation;
3. Training and evalution of model with different types of metadata.

The application was developed in Python 3.6 and is based on four libraries: LightFM - is used to build the recommender system; Keras - is used for feature extraction from the movies posters; Scikit learn - is used to create clusters; Skopt - is used to optimize the model parameters. The application was run on Intel i7 Quad Core, 2.60 Ghz with 16GB RAM and with a swap memory extension up to 70GB.

From the point of view of results we got an improvement of the precision@k and accuracy metrics. If is used just the movies posters, precision is improvement with 0.42\% and accuracy with 0.32\%. If posters is used together with genres, precision is improvement with 0.82\% and accuracy with 1.09\%.

The paper can be found [here](https://github.com/adiIspas/King-Recommender-Paper).

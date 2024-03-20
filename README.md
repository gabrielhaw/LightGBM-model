# LightGBM-model
The provided code was utilised to test a LightGBM-model in terms of predictive accuracy on a methylation dataset. This model was trained to predict the relative age given the methylation profiles of various cytosine-guanine (cg) sites. For this model we utilised the dataset coming from the [DNA methylation networks underlying mammalian traits](https://www.science.org/doi/10.1126/science.abq5693) paper by Amin Haghani et al. 

Please note the model was trained on the double negative log age in order to stabilise for variance within the dataset, a procedure followed from the paper, and was then converted back into relative age using the appropriate mathematical transformations. 

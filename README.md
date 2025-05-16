This is the repo for a project done at NEUR240, Biological and Artificial Intelligence (Harvard, Spring 2025), with title:

# Is HMAX, a biologically plausible model, subject to the Müller–Lyer illusion?

Here, I investigate whether a biologically inspired vision pipeline—combining the HMAX hierarchical 
feature extractor (Riesenhuber & Poggio, 1999) with a linear support vector machine (Cortes & 
Vapnik, 1995)—naturally exhibits the Müller–Lyer illusion despite being trained only on 
veridical length discrimination. First, I validated the model’s pure length‐comparison ability on a 
Control-Figure (CF) dataset by sweeping fin angle, fin length, shaft length, and vertical 
separation; performance remained uniformly high (>85 %) across all but the most extreme fin 
geometries, confirming robust CF classification. I then applied the trained classifier to a Müller–
Lyer (ML) test set and observed a systematic bias: classification accuracy plunged to ~30 % 
when arrowheads pointed in opposite directions, well below chance, but recovered to ~82 % 
under uniform arrow direction. Further parametric analysis revealed that fin angle and shaft 
length are the principal determinants of illusion strength—mid-range angles and short shafts 
maximize misperception—while alterations of fin length or line separation have negligible 
impact. These findings mirror human psychophysics and demonstrate that simple, feedforward 
feature hierarchies can inherit core perceptual biases, offering a powerful framework for probing 
the mechanistic origins of visual illusions in both artificial and biological systems. 

# Logistics
The repo is a little bit unorganized, but users can replicate my finding by running `official.ipynb`. Directories inside the code should be adjusted accordingly if running on your local computer. 

`data_generation.py`: How the training and testing datasets are created. You can change configurations of the data accordingly. \
`dataset.ipynb`: Run this to create the data \
`Dataset.py`: Dataset classes for PyTorch integration 

The implementation of HMAX was adapted from [github.com/wmvanvliet/pytorch_hmax/](https://github.com/wmvanvliet/pytorch_hmax/)

### You can read my report at `Paper.pdf`.

## References
Riesenhuber, M., & Poggio, T. (1999). Hierarchical models of object recognition in cortex. Nature Neuroscience, 
2(11), 1019–1025. https://doi.org/10.1038/14819
Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297. 
https://doi.org/10.1007/BF00994018

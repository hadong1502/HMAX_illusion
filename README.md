This is the repo for a class project with title "Is HMAX, a biologically plausible model, subject to the Müller–Lyer illusion?". 

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

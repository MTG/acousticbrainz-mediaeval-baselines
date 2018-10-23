MediaEval 2018 AcousticBrainz Genre Task
========================================
A baseline combining deep feature embeddings across datasets
------------------------------------------------------------

This is the code for our baseline submission to the MediaEval 2018 AcousticBrainz Genre Task. 

The goal of the task is to automatically classify music tracks by genres based on pre-computed audio
content features provided by the organizers. Four different genre datasets coming from different 
annotation sources with different genre taxonomies are used in the challenge. For each dataset, 
training, validation, and testing splits are provided. This allows to build and evaluate classifier
models for each genre dataset independently (Task 1) as well as explore combinations of genre sources 
in order to boost performance of the models (Task 2). 

In this baseline, we decided to focus on demonstration of possibilities of merging different genre 
ground truth sources using a simple deep learning architecture.

TODO

### Running code
TODO


### License
Source code and models can be licensed under the GNU AFFERO GENERAL PUBLIC LICENSE v3.
For details, please see the `LICENSE <LICENSE>`_ file.


### Citation

If you use this project in your work, please consider citing this publication:

```latex
@inproceedings{
    Title = {Media{E}val 2018 Acoustic{B}rainz Genre Task: A Baseline Combining Deep Feature Embeddings Across Datasets},
    Author = {Oramas, Sergio and Bogdanov, Dmitry and Porter, Alastair},
    Booktitle = {Proceedings of the Media{E}val 2018 Multimedia Benchmark Workshop},
    Month = {10},
    Year = {2018},
    Address = {Sophia Antipolis, France}
}
```

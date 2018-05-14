# Classification of bioacoustic signals using unsupervised learning techniques

## Synopsis

This repository contains both the report and implementation of a Bachelor Thesis from the University of Havana, Cuba. Its contents are distributed across several folders using the following layout:

```text
clusterapp/ // Source code of the project implementation 
codes/      // Source code using to generate plots and results shown in the thesis report
sounds/     // Datasets of sounds used in the research
thesis/     // LaTeX source code of the thesis report (Please be aware the report is written in Spanish)
```

In order to generate a pdf file from the _LaTeX_ source code it is recommended to use the _MiKTeX_ distribution. The following commands have to be executed:

```bash
pdflatex thesis/title.tex
pdflatex thesis/main.tex
bibtex thesis/main.tex
pdflatex thesis/main.tex
pdflatex thesis/main.tex
```

## Clusterapp

_Clusterapp_ is a Python module which provides a library for clustering processing of bioacoustic sounds. Interaction with it can be done from three separate ways:

### Web

To use the module this way, the following command has to be issued from a terminal:

```bash
python -m clusterapp <path_to_dataset> [--classified] web [--host <host_address>, --port <port_number>]
```
Optional arguments:

* `--classified`: Indicates whether or not the names of files in dataset should be considered as their classification (See example below).
* `--host`: IP address or name of host where web server is being deployed.
* `--port`: Port number by which it will be accessible.

Datasets should be composed by a folder only composed by sound files (currently in .WAV format only). In order to use the dataset with the `--classified` argument, all of its files need to be named according with the following pattern:
`<name_of_category>-<file_id>.wav`, where `<name_of_category>` is the name of the type of vocalization by which the segment in the file is classified is identified (e.g. _cannis lupus A_).

 Valid patterns are, for example, _cannis luppus A-1.wav_, _cannis lupus B-2.wav_ and _nyctalus noctula-1.wav_. In this case a letter was added at the end of the species name of the first two files, in order to distinguish different types of vocalizations coming from the same species.

Once the server is running a message will be printed in the terminal indicating the address by which it can be accessed from a web browser (default is http://localhost:5000).

#### Dataset Analysis

This is the first view of the two that compose the web interface. Dataset Analysis allows the user to explore the results produced by different combinations of clustering algorithms and features in the selected categories of sounds.

Two options are presented at first:

* __Clustering algorithm__: Use it to select the algorithm to work with. If the application is running on a `--classified` dataset, there will be an extra one called `None`, that will group files by their true clategories (i.e. it will not take features into account).
* __Features selector__: Use it to choose the combination of features from segments to analyse.

A third option varies according to if the dataset is `--classified` or not:
* __Filter by species__: For `--classified` datasets. Use it to select which categories from the dataset will be analysed.
* __Number of clusters__: As there is no information regarding the number of categories present in non-classified datasets, some algorithms may request the user to manually introduce the number of clusters they expect.

Once everything is configured as desired, and after pressing the _Refresh_ button, the result will be reflected in a scatter chart and a table, presenting statistics and composition of clusters. In case of `--classified` datasets, the table will contain more information regarding composition of clusters in comparison with the true categories of data.

How the scatter chart is constructed varies according to the number of features selected. In cases of only two features it will be made by corresponding each axis to one feature, which will be indicated in the chart. However if the number of features is different than two, the conversion of the vector of features of each point to two dimensions will be done by using [Multi-dimensional Scaling](https://en.wikipedia.org/wiki/Multidimensional_scaling), in order to show a 2D approximation of their distributions in their original space.

Additionally if the dataset is `--classified`, a new option will be presented after pressing the _Refresh_ button, allowing the user to load a new audio file which will be processed and classified according to the cluster with the nearest center. The new point will be also included in the scatter chart.

#### Best features

This view performs a selection of the set of features for which the selected clustering method produces its best result. Two options are first presented: One for selecting the clustering algorithm to use, and other for choosing the sizes of the sets of features to test. Results can be better for larger intervals, but the calculation will take longer.

A third option will vary according to whether the dataset is `--classified` or not. In the first case, it will allow to choose the categories to use for the selection. In the other case, the selection will be done using all the files of the dataset, as there is no way to distinguish between them, and for some algorithms the number of clusters will be asked for.

After pressing the _Refresh_ button the calculations (which might take a while) will be performed, and their results will appear in a scatter chart and a table, similar to those of the _Dataset Analysis_ section.

### Command Line Interface

### Importing from Python code
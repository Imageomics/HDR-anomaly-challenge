# Butterfly Hybrid Detection
### Brought to you by Imageomics Institute as part of the 2024 HDR Anomaly ML Challenge


## Introduction 

### Butterfly hybrids
It is known that a butterfly species may develop into multiple subspecies according to the habitat regions. The visual appearances (e.g., color patterns on the wings) of these subspecies can be drastically different.

Normally, the same subspecies mate and produce children. Occasionally, different subspecies would mate and produce children that are considered **hybrids**, whose visual appearances are partially similar to each of their parents.

In this challenge, children produced by the same-subspecies parents (i.e., **non-hybrids**) are treated as **normal** cases because they are much more frequently observed. In contrast, **hybrids** are treated as **anomaly** cases, not only because they are much less frequently observed--with some combinations not yet observed--but also because their visual appearances are much more variant and hardly predictive.

### Butterfly mimicry
On a different dimension, two different butterfly species with overlapping habitats may visually mimic each other. Such mimicry could help avoid shared predators, especially if one (or both) species is toxic or not palatable.

Besides the goal to develop an anomaly detection algorithm to distinguish between hybrids and non-hybrids for one species, this challenge aims to investigate whether such an algorithm is generalizable to the other visually mimicking species. 

![subspecies_to_hybrids comparison different subspecies of Species A compared to two of its mimics](https://github.com/user-attachments/assets/8647e1f5-4f99-48c6-8325-fdfd0e5d4c21)


## Setup Overview
This challenge is designed to simulate a real-world biological scenario. Suppose a biologist studies a particular butterfly Species A. One day, the biologist finds that a subset of the images collected looks slightly abnormal in their visual appearance. After investigation, the biologist finds that these abnormal samples are hybrids produced by different subspecies of Species A, which are rarely observed. Since in theory, there are quadratically many possibilities of hybrids and the current collection only covers a small subset of them, the biologist seeks an anomaly detection algorithm to automatically identify (unseen) hybrid cases from future image collections of Species A.

### Algorithm requirement
The developed anomaly detection algorithm needs to output an anomaly score (a real number) for each test image. The higher the score is, the more likely the image is an anomaly (i.e., hybrid).

### Training data
The training data comprises images from all the Species A subspecies and the most common hybrid. The most common* hybrid refers to a specific combination of the parent subspecies that has the most images. This hybrid is called the signal hybrid; other hybrids are called the non-signal hybrids.    

### Test data
We consider two sets of images in the test set:
- One from Species A, comprising images from all the Species A subspecies, the signal hybrid, and the non-signal hybrids.
- One from a mimicking Species B, comprising images from two subspecies of Species B and their hybrid. The two subspecies of Species B are the ones mimicking the parent subspecies of the signal hybrid of Species A.

## Timeline

This ML Challenge starts on September [start day], 2024, and will run through [end date]. Be sure to resubmit your preferred model from the development phase by [date, time]; it will then be run on the final test sets. Only one submission will be run against the test sets to determine your final score in the challenge.


\*  Note that these hybrids are just the most common within this particular dataset, not necessarily in general.

**References and credits:** Zenodo citations.<br />
This challenge was generated using the [CodaBench Iris Sample Bundle](https://github.com/codalab/competition-examples/tree/master/codabench/iris/bundle).

# Butterfly Hybrid Detection
### Brought to you by Imageomics Institute as part of the 2024 HDR Anomaly ML Challenge


## Introduction 

### Butterfly hybrids
Populations of butterfly species that become separated in some way, such as geographic or habitat separation, can develop into different subspecies. The visual appearances (e.g., color patterns on the wings) of these subspecies can be drastically different.

Normally, only individuals of the same subspecies mate and produce offspring. Occasionally, where the ranges of subspecies come into contact or overlap, individuals from different subspecies can mate, and produce offspring considered **hybrids**. The visual appearances of hybrids are _partially_ similar to _each_ of their parents.

In this challenge, offspring produced by the same-subspecies parents (i.e., **non-hybrids**) are treated as **normal** cases because they are far more frequently observed. In contrast, **hybrids** are treated as **anomaly** cases, not only because they are much less frequently observed--with some combinations not yet observed--but also because their visual appearances are much more variant and hardly predictive.

### Butterfly mimicry
On a different dimension, two different butterfly species with overlapping geographic ranges may visually mimic each other. Such [mimicry](https://en.wikipedia.org/wiki/M%C3%BCllerian_mimicry) could help avoid shared predators, especially if one (or both) species is toxic or not palatable.

Besides the goal to develop an anomaly detection algorithm to distinguish between hybrids and non-hybrids for one species, this challenge aims to investigate whether such an algorithm is generalizable to the other visually mimicking species. 

![subspecies_to_hybrids comparison different subspecies of Species A compared to two of its mimics](https://github.com/user-attachments/assets/8647e1f5-4f99-48c6-8325-fdfd0e5d4c21)


## Setup Overview
This challenge is designed to simulate a real-world biological scenario. Suppose a biologist studies a particular butterfly Species A with many subspecies. One day, the biologist finds that a subset of the images collected looks slightly abnormal in their visual appearance. The biologist does not recognize the pattern as belonging to any of the subspecies on which their research is focused. After investigation, the biologist finds that these unusual samples are hybrids produced by different subspecies of Species A. Realizing that they may encounter other hybrids in future collections of images, the biologist seeks an anomaly detection algorithm to automatically identify (unseen) hybrid cases.

### Algorithm requirement
The developed anomaly detection algorithm needs to output an anomaly score (a real number) for each test image. The higher the score is, the more likely the image is an anomaly (i.e., hybrid).

### Training data
The training data comprises images from all the Species A subspecies and the most common hybrid. The most common* hybrid refers to a specific combination of the parent subspecies that has the most images. This hybrid is called the signal hybrid; other hybrids are called the non-signal hybrids.    

### Test data
We consider two sets of images in the test set:
- One from Species A, comprising images from all the Species A subspecies, the signal hybrid, and the non-signal hybrids.
- One from a mimicking Species B, comprising images from two subspecies of Species B and their hybrid. The two subspecies of Species B are the ones mimicking the parent subspecies of the signal hybrid of Species A.

## Timeline

This ML Challenge starts on September 11th, 2024, and will run through January 17th, 2025. Be sure to resubmit your preferred model from the development phase by January 17th at 11:59pm [AOE](https://www.timeanddate.com/time/zones/aoe); it will then be run on the final test sets. Only one submission will be run against the test sets to determine your final score in the challenge.


\*  Note that these hybrids are just the most common within this particular dataset, not necessarily in general.

**References and credits:** [Zenodo citations](https://github.com/Imageomics/HDR-anomaly-challenge/blob/main/butterfly_anomaly.bib).<br />
This challenge was generated based on the [CodaBench Iris Sample Bundle](https://github.com/codalab/competition-examples/tree/master/codabench/iris/bundle); full formatting and challenge design process can be found in the [challenge repo on GitHub](https://github.com/Imageomics/HDR-anomaly-challenge).

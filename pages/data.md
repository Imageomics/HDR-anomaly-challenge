# Butterfly Hybrid Detection: Data

This challenge uses data compiled from various [Zenodo records of the Butterfly Genetics Group at University of Cambridge](https://zenodo.org/communities/butterfly/records?q=&f=subject%3ACambridge&l=list&p=1&s=10&sort=newest). All data is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/); see [butterfly_anomaly.bib](https://github.com/Imageomics/HDR-anomaly-challenge/blob/main/butterfly_anomaly.bib) for full citations.

## Instructions to Download Training Data

First, install the downloader in your virtual environment:
```bash
pip install git+https://github.com/Imageomics/cautious-robot
```
Then download [`butterfly_anomaly_train.csv`](https://github.com/Imageomics/HDR-anomaly-challenge/blob/main/files/butterfly_anomaly_train.csv) and run: 
```bash
cautious-robot -i <path/to/butterfly_anomaly_train.csv> -o <path/to/images> -s hybrid_stat -v md5
```

This will create subfolders `hybrid` and `non-hybrid` with images named by the `filename` column (`<CAMID>.jpg`). Remove the `-s hybrid_stat` if you want a flat directory.

Add downsample flag with desired size if you want to also get the images downsized for training (e.g., `-l 224` for 224 x 224 images). This creates a directory `path/to/images_downsized` with the downsized images in the same folder structure as the originals.

`-v md5` will compare the checksum file with the checksums in the provided training data CSV to ensure all images were downloaded correctly; check the download logs if any are missing (see [cautious-robot](https://github.com/Imageomics/cautious-robot) for more information on download options).


## Additional Information About the CSV File

Following the above steps, participants will obtain two training image subfolders, one for **hybrid** and one for **non-hybrid**. 

The [`butterfly_anomaly_train.csv`](https://github.com/Imageomics/HDR-anomaly-challenge/blob/main/files/butterfly_anomaly_train.csv) offers additional biologically meaningful information for each image, which may be useful for developing the anomaly (i.e., hybrid) detection algorithm.

- Column **subspecies**: the subspecies of each **non-hybrid** image. For images in the **hybrid** subfolder, this information is empty.
- Columns **parent_subspecies_1** and **parent_subspecies_2**: the parent subspecies of each **hybrid** image. For images in the **non-hybrid** subfolder, this information is empty.

It is worth noting that in the **hybrid** subfolder, only the signal hybrid from one specific combination of parent subspecies is provided for training. However, in the test set, hybrids from other combinations of parent subspecies will be included. Namely, not all possible hybrid cases are observed in the training set, consistent with the challenge of anomaly detection. 


## Submission Samples

Participants can download sample submissions with the baseline algorithms (`DinoV2` and `BioCLIP` based) from the "Files" tab.

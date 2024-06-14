# Butterfly Hybrid Detection: Data

This challenge uses data compiled from various Zenodo records of the Butterfly Genetics Group. [More Info Coming]

## Instructions to Download Training Data

First, install the downloader in your virtual environment:
```bash
pip install git+https://github.com/Imageomics/cautious-robot.git@v0.1.1-alpha
```
Then download `butterfly_anomaly_train.csv` from the "Files" tab and run: 
```bash
cautious-robot -i <path/to/butterfly_anomaly_train.csv> -o <path/to/images> -s hybrid_stat -n CAMID -v md5
```

This will create subfolders `hybrid` and `non-hybrid` with images named by the `CAMID` column. Remove the `-s hybrid_stat` if you want a flat directory.

Add downsample flag with desired size if you want to also get the images downsized for training (e.g., `-l 224` for 224 x 224 images). This creates a directory `path/to/images_downsized` with the downsized images in the same folder structure as the originals.

`-v md5` will compare the checksum file with the checksums in the provided training data CSV to ensure all images were downloaded; check the download logs if any are missing (see [cautious-robot](https://github.com/Imageomics/cautious-robot) for more information on download options).

## Submission Samples

Participants can download sample submissions with the baseline algorithms (`DinoV2` and `BioCLIP` based) from the "Files" tab.

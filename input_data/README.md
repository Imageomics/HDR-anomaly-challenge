This folder should have `input_data_species_A` and `input_data_mimic` for training, validation, and testing, where validation is the unlabeled dataset for the "Development Task" on which participants may receive feedback.

I think this would actually be the images (or the ingestion program would use the data provided here to get the images) if we are taking models as submissions and running them to get the predictions.

In the [Iris example](https://github.com/codalab/competition-examples/tree/master/codabench/iris/bundle/input_data), this has the unlabeled data for each of validation and testing, as well as training data and training data labels.
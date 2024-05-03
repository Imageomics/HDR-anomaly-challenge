This folder should have the following data
 - `valid_A`: Validation set for Species A hybrid detection.
 - `test_A`: Test set for Species A hybrid detection.
 - `valid_mimic`: Validation set for Mimic hybrid detection.
 - `test_mimic`: Test set for Mimic hybrid detection.

I think this would just be the labels if we're looking at assessing their predictions. May be labels regardless: in the [Iris example](https://github.com/codalab/competition-examples/tree/master/codabench/iris/bundle/reference_data), this is the set of labels in folders "valid" and "test", while the [`input_data` folder](https://github.com/codalab/competition-examples/tree/master/codabench/iris/bundle/input_data) has the unlabeled data for each of validation and testing, as well as training data and training data labels.
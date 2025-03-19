This folder contains the following data which was kept secret for the duration of the challenge
 - `ref_val.csv`: Validation set used during the Development Phase. It includes both Species A & B, as well as subspecies hybrids of both, beyond just the Signal Hybrids.
 - `ref_test.csv`: Test set used for final evaluation of submissions. It includes both Species A & B, as well as subspecies hybrids of both, beyond just the Signal Hybrids.

> [!NOTE]  
> - Both of these CSVs contain a `split` column with both `test` and `test-2`; this was an artifact of our original pre-competition setup that distinguished multiple tests within each phase (one for Species A hybrid detection and one for Species B hybrid detection). 
> - Actual subspecies epithets are provided in `subspecies_ref`, those with `ssp_indicator` mimic are _Heliconius melpomene_ and the rest are _Heliconius erato_.
> - `X` is a unique identifier for all the images used in this challenge, as is `CAMID`.
> - Images may be downloaded following the [training data download instructions](../pages/data.md#instructions-to-download-training-data).

# Evaluating Zero-Shot CLIPSeg Models on Indoor Object Segmentation

## Group - Roland Gulbinovič & Danial Yntykbay

Dataset used - https://github.com/apple/ml-hypersim

Models compared:
- rd64-refined
- rd64
- rd16

Classes used for testing and evaluation - "Bed", "Table", "Chair", "Floor".


### Zero-shot Evaluation of CLIPSeg Variants

| Model        | Prompt   | F1      | Accuracy | Sensitivity | Specificity |
|--------------|----------|---------|----------|-------------|-------------|
| rd64-refined | the floor| 0.721299| 0.938724 | 0.698056    | 0.974041    |
| rd64-refined | a chair  | 0.457096| 0.949868 | 0.560720    | 0.967503    |
| rd64-refined | a bed    | 0.537657| 0.938838 | 0.530287    | 0.974745    |
| rd64-refined | a table  | 0.283398| 0.956490 | 0.324818    | 0.971355    |
| rd64         | the floor| 0.717041| 0.935251 | 0.678875    | 0.972882    |
| rd64         | a chair  | 0.409269| 0.948246 | 0.474542    | 0.971384    |
| rd64         | a bed    | 0.507191| 0.939282 | 0.509004    | 0.973872    |
| rd64         | a table  | 0.273899| 0.953391 | 0.366174    | 0.968455    |
| rd16         | the floor| 0.704214| 0.934163 | 0.663723    | 0.971526    |
| rd16         | a chair  | 0.365508| 0.944872 | 0.430006    | 0.968466    |
| rd16         | a bed    | 0.553340| 0.941451 | 0.595965    | 0.957202    |
| rd16         | a table  | 0.222279| 0.947818 | 0.323232    | 0.962789    |

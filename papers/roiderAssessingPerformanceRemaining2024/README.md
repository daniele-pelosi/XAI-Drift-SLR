# Assessing the Performance of Remaining Time Prediction Models for Business Processes

## Prepare Data

Running the file ```prepare_datasets.py [dataset] [input dataset location] [prefixes location]``` performs the following steps:
- Split dataset into training, validation, and test sets
- Calculate timestamp related features
- Prepare prefix data for PGT-Net and save to disk
- Prepare prefix data for DA-LSTM and save to disk

The input parameters are as follows:
```
- dataset: Name of the event log, excluding file extension
- input dataset location: General information like the data split are stored into this folder
- prefixes location: Folder to which the prefixes for training are saved.
```

Example:
```python prepare_datasets.py bpic2015_1 data/datasets/ data/preprocessed/```

## Run Experiments

To start experiment, run the following command:
```python main.py [dataset] [model_type] [seed]```

```
- dataset: dataset name (e.g., bpic2015_1, helpdesk, ...)
- model_type: model type (e.g., dalstm; xgboost: xboost using square error criterion; xgboostl1: xgboost using absolute 
error criterion; graphgps: PGT-Net)
- seed: random seed
```

Example:
```python main.py bpic2015_1 dalstm 0```


## Datasets

### bpic2012a

The first activity must always be ```A_SUBMITTED```. 

Cases can either be successful or not. For cases not successful, the last activity must be either ```A_DECLINED``` or 
```A_CANCELLED```. 

If a case is successful, it must contain the activities ```A_APPROVED```, ```A_REGISTERED```, and ```A_ACTIVATED```. 
At the same time it is not allowed to contain the activities ```A_DECLINED``` or ```A_CANCELLED```.


### bpic2012o

The first activity of a case must always be ```O_SELECTED```.

The last activity of a case must be one of: ```O_DECLINED```, ```O_CANCELLED```, ```O_ACCEPTED```.


### bpic2020_dd

The first activity of a case must always be ```Declaration SUBMITTED by EMPLOYEE```.

The last activity of a case must be one of: ```Payment Handled```, ```Declaration REJECTED by EMPLOYEE```.


### bpic2020_id

The first activity must be one of: ```'Start trip'```, ```Declaration SUBMITTED by EMPLOYEE```, 
```Permit SUBMITTED by EMPLOYEE```.

The last activity must be one of: ```End Trip```, ```Payment Handled```, ```Declaration REJECTED by EMPLOYEE```.


### bpic2020_ptc

A case can start with any activity.

The last activity of a case must be one of: ```Payment Handled```, ```Request For Payment REJECTED by EMPLOYEE```.


### bpic2020_rp

The first activity must be ```Request For Payment SUBMITTED by EMPLOYEE```.

The last activity of a case must be one of: ```Payment Handled```, ```Request For Payment REJECTED by EMPLOYEE```.


### credit

The first activity must be ```Register```.

The last activity must be ```Requirements review```.


### helpdesk

The fist activity must be one of: ```Assign Seriousness``` or ```Insert ticket```

The last activity must be always ```closed```. 

```Assign Seriousness``` must always occur at least once in a case.


### hospital

The first activity must be ```NEW```.

The last activity must be one of: ```BILLED```, ```DELETE```.


### sepsis

The first activity must be ```ER Registration```.

The last activity must be one of: ```RELEASE_A```, ```Discharge_B```, ```Discharge_C```, ```Discharge_D```, 
```Discharge_E```, ```Discharge_F```, ```Return ER```.

If ```RELEASE_A``` occurs for a case, none of the following is allowed to occur for the same case: ```Discharge_B```, ```Discharge_C```, 
```Discharge_D```, ```Discharge_E```.

If ```Discharge_B``` occurs for a case, none of the following is allowed to occur for the same case: ```RELEASE_A```, ```Discharge_C```, 
```Discharge_D```, ```Discharge_E```.

If ```Discharge_C``` occurs for a case, none of the following is allowed to occur for the same case: ```RELEASE_A```, ```Discharge_B```, 
```Discharge_D```, ```Discharge_E```.

If ```Discharge_D``` occurs for a case, none of the following is allowed to occur for the same case: ```RELEASE_A```, ```Discharge_B```, 
```Discharge_C```, ```Discharge_E```.

If ```Discharge_E``` occurs for a case, none of the following is allowed to occur for the same case: ```RELEASE_A```, ```Discharge_B```, 
```Discharge_C```, ```Discharge_D```.
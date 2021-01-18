# Layout IPA - Using Layout structures for Intelligent Process Automation

TODO: Add proper description.

## Installing the requirements

```
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

In order for the cache mechanism to work, run the following:

```
export PREFECT__FLOWS__CHECKPOINTING=true
```

or add it to .bashrc:

```
echo 'export PREFECT__FLOWS__CHECKPOINTING=true' >> ~/.bashrc 
```

## Running Baselines

### BERT

#### Pair Classification

```
python -m  layout_ipa.flows.transformers_based.transformers_train_pair_classification
```

#### Element Selection


#### Embedding


### LayoutLM

#### Pair Classification


#### Element Selection


#### Embedding


### Running the flows

#### Pair Classification


#### Element Selection


#### Embedding



Running the example flow:

```
python -m  sample.flows.sample_flow
```







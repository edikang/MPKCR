# MPKCR: Multi-level Preference Modeling with Knowledge-enhanced for Conversational Recommendation

## Requirements
```
python==3.8.12
pytorch==1.10.1
dgl==0.4.3
cudatoolkit==10.2.89
torch-geometric==2.0.3
transformers==4.15.0
```

## Running
```
cd ../MPKCR
python run_crslab.py --config config/crs/mpkcr/hredial.yaml -g 0 -s 1 -p -e 10
python run_crslab.py --config config/crs/mpkcr/htgredial.yaml -g 0 -s 1 -p -e 10
```

## Acknowledgement
The implementation is based on the open-source CRS toolkit [CRSLab](https://github.com/RUCAIBox/CRSLab).

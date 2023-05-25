![readme_banner](https://user-images.githubusercontent.com/81975877/225590148-0c5d9c19-8e66-4b07-855b-0fd6dfb059eb.jpg)
*AWESAM - **A**daptive-**W**indow Volcanic **E**vent **S**election **A**nalysis **M**odule*

> Module for creating seismo-volcanic event catalogs from seismic data for volcanoes with frequent activity.

See https://doi.org/10.3389/feart.2022.809037 for a detailed description. The process consists of three steps:

1. **EventDetection:** Identification of potential volcanic events based on squared ground-velocity amplitudes, an adaptive MaxFilter, and a prominence threshold. 
2. **CatalogConsolidation:** By comparing and verifying the initial detections based on recordings from two different seismic stations. 
3. **EarthquakeClassification:** Identification of signals from regional tectonic earthquakes (based on an earthquake catalog)

## Tutorial

- Tutorial Notebook [AwesamTutorial.ipynb](AwesamTutorial.ipynb)
- Tutorial Video [AWESAM Tutorial (Youtube)](https://www.youtube.com/watch?v=S3OQ3mT96CY)

## Dependencies
When using awesamlib (after compilation with e.g. gcc): `numpy 1.21.5`, `obspy 1.3.0`, `scipy 1.8.0`, `pandas 1.4.1`, `torch 1.11.0`. When using the python-backend, additionally `numba 0.55.1` is needed. Developed with `python 3.8.10`.
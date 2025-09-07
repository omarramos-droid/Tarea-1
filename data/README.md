
## Data source

The data came from [GFZ Data Services](https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=082517c6-a951-11ed-95b8-f851ad6d1e4b)

## How to use it?


From the parent directory you just need to run
```bash 
python -m scripts.get_raw_data
```

It will download the zip file from the [url](https://datapub.gfz-potsdam.de/download/10.5880.GFZ.4.3.2023.002sdfsd/2023-001_ISONET-Project-Members_13C_Data.zip) and unzip it then put it in the `data/raw` directory.

If the [url](https://datapub.gfz-potsdam.de/download/10.5880.GFZ.4.3.2023.002sdfsd/2023-001_ISONET-Project-Members_13C_Data.zip) changes, then we have to modify the the global variable `data_url` from the `scripts/config.py` file to point to the new corrected address (luckily this will change).

## References
ISONET Project Members; Schleser, Gerhard Hans; Andreu-Hayles, Laia; Bednarz, Zdzislaw; Berninger, Frank; Boettger, Tatjana; Dorado-Liñán, Isabel; Esper, Jan; Grabner, Michael; Gutiérrez, Emilia; Helle, Gerhard; Hilasvuori, Emmi; Jugner, Högne; Kalela-Brundin, Maarit; Krąpiec, Marek; Leuenberger, Markus; Loader, Neil J.; Masson-Delmotte, Valérie; Pawełczyk, Sławomira; Pazdur, Anna; Pukienė, Rūtilė; Rinne-Garmston, Katja T.; Saracino, Antonio; Saurer, Matthias; Sonninen, Eloni; Stiévenard, Michel; Switsur, Vincent R.; Szychowska-Krąpiec, Elżbieta; Szczepanek, M.; Todaro, Luigi; Treydte, Kerstin; Vitas, Adomas; Waterhouse, John S.; Weigl-Kuska, Martin; Wimmer, Rupert (2023): Stable carbon isotope ratios of tree-ring cellulose from the site network of the EU-Project ‘ISONET’. GFZ Data Services. https://doi.org/10.5880/GFZ.4.3.2023.002
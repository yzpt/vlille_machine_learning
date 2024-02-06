gcloud config list

gcloud projects list
# set project vlille-gcp-etl
gcloud config set project vlille-gcp-etl

bq ls
# get the region and/or location of the dataset
bq show --format=prettyjson vlille-gcp-etl:vlille_dataset
# location US

# create a new dataset vlille_ml_test
bq --location=US mk --dataset vlille-gcp-etl:vlille_ml_test

# copy the data from vlille_dataset.stations to vlille_ml_test.stations
bq cp vlille-gcp-etl:vlille_dataset.stations vlille-gcp-etl:vlille_ml_test.stations

# copy the data from vlille_dataset.records to vlille_ml_test.records
bq cp vlille-gcp-etl:vlille_dataset.records vlille-gcp-etl:vlille_ml_test.records

# assuming the json schema of the records table:
[
    {"name": "station_id",          "type": "INT64"},
    {"name": "etat",                "type": "STRING"},
    {"name": "nb_velos_dispo",      "type": "INT64"},
    {"name": "nb_places_dispo",     "type": "INT64"},
    {"name": "etat_connexion",      "type": "STRING"},
    {"name": "derniere_maj",        "type": "TIMESTAMP"},
    {"name": "record_timestamp",    "type": "TIMESTAMP"}
]

# remove duplicates from the records table
bq query --use_legacy_sql=false 'CREATE OR REPLACE TABLE vlille-gcp-etl.vlille_ml_test.records AS SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY station_id, record_timestamp ORDER BY derniere_maj DESC) AS rn FROM vlille-gcp-etl.vlille_ml_test.records) WHERE rn = 1'


# featuring the records table
bq query --use_legacy_sql=false --destination_table=vlille-gcp-etl:vlille_ml_test.test_ml_3 '
SELECT
    station_id,
    EXTRACT(DAYOFWEEK FROM record_timestamp) AS day_of_week,
    EXTRACT(HOUR FROM record_timestamp) AS hour_of_day,
    IF(nb_velos_dispo + nb_places_dispo > 0, nb_velos_dispo / (nb_velos_dispo + nb_places_dispo), NULL) AS ratio_velos_dispo
FROM
    `vlille-gcp-etl.vlille_ml_test.records`
WHERE
    station_id = 25
'


bq extract --destination_format=CSV vlille-gcp-etl:vlille_ml_test.test_ml_3 gs://vlille-gcp-etl-data/test_ml_3.csv

gsutil cp gs://vlille-gcp-etl-data/test_ml_3.csv test_ml_3.csv

# --> test_ml_3.csv

git init
git add .
git commit -m "First commit"
git remote add origin https://github.com/yzpt/vlille_machine_learning.git
git push -u origin main
COPY carcinogenesis."atom" FROM '/data/carcinogenesis/atom.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY carcinogenesis."canc" FROM '/data/carcinogenesis/canc.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY carcinogenesis."sbond_1" FROM '/data/carcinogenesis/sbond_1.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY carcinogenesis."sbond_2" FROM '/data/carcinogenesis/sbond_2.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY carcinogenesis."sbond_3" FROM '/data/carcinogenesis/sbond_3.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY carcinogenesis."sbond_7" FROM '/data/carcinogenesis/sbond_7.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;

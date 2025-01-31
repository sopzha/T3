COPY seznam."client" FROM '/data/seznam/client.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY seznam."dobito" FROM '/data/seznam/dobito.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY seznam."probehnuto" FROM '/data/seznam/probehnuto.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY seznam."probehnuto_mimo_penezenku" FROM '/data/seznam/probehnuto_mimo_penezenku.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;

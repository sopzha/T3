COPY accident."nesreca" FROM '/data/accident/nesreca.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY accident."oseba" FROM '/data/accident/oseba.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY accident."upravna_enota" FROM '/data/accident/upravna_enota.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;

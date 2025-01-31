COPY hepatitis."Bio" FROM '/data/hepatitis/Bio.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY hepatitis."dispat" FROM '/data/hepatitis/dispat.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY hepatitis."indis" FROM '/data/hepatitis/indis.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY hepatitis."inf" FROM '/data/hepatitis/inf.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY hepatitis."rel11" FROM '/data/hepatitis/rel11.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY hepatitis."rel12" FROM '/data/hepatitis/rel12.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY hepatitis."rel13" FROM '/data/hepatitis/rel13.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;

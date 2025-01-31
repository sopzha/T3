COPY fhnk."pripady" FROM '/data/fhnk/pripady.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY fhnk."vykony" FROM '/data/fhnk/vykony.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY fhnk."zup" FROM '/data/fhnk/zup.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;

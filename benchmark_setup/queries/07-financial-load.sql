COPY financial."account" FROM '/data/financial/account.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY financial."card" FROM '/data/financial/card.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY financial."client" FROM '/data/financial/client.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY financial."disp" FROM '/data/financial/disp.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY financial."district" FROM '/data/financial/district.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY financial."loan" FROM '/data/financial/loan.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY financial."order" FROM '/data/financial/order.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY financial."trans" FROM '/data/financial/trans.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;

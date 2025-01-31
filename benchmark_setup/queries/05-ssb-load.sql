COPY ssb.customer FROM '/data/ssb/customer.csv' DELIMITER '|' QUOTE '"' ESCAPE '\' NULL '' CSV HEADER;
COPY ssb.part FROM '/data/ssb/part.csv' DELIMITER '|' QUOTE '"' ESCAPE '\' NULL '' CSV HEADER;
COPY ssb.supplier FROM '/data/ssb/supplier.csv' DELIMITER '|' QUOTE '"' ESCAPE '\' NULL '' CSV HEADER;
COPY ssb.lineorder FROM '/data/ssb/lineorder.csv' DELIMITER '|' QUOTE '"' ESCAPE '\' NULL '' CSV HEADER;
COPY ssb.dim_date FROM '/data/ssb/dim_date.csv' DELIMITER '|' QUOTE '"' ESCAPE '\' NULL '' CSV HEADER;

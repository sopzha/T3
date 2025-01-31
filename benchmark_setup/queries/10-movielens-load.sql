COPY movielens."actors" FROM '/data/movielens/actors.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY movielens."directors" FROM '/data/movielens/directors.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY movielens."movies" FROM '/data/movielens/movies.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY movielens."movies2actors" FROM '/data/movielens/movies2actors.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY movielens."movies2directors" FROM '/data/movielens/movies2directors.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY movielens."u2base" FROM '/data/movielens/u2base.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY movielens."users" FROM '/data/movielens/users.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;

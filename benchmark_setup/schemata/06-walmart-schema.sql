CREATE SCHEMA walmart;
DROP TABLE IF EXISTS walmart."key";

CREATE TABLE walmart."key"
(
    "store_nbr"   integer,
    "station_nbr" integer DEFAULT NULL,
    PRIMARY KEY ("store_nbr")
);

DROP TABLE IF EXISTS walmart."station";

CREATE TABLE walmart."station"
(
    "station_nbr" integer,
    PRIMARY KEY ("station_nbr")
);

DROP TABLE IF EXISTS walmart."train";

CREATE TABLE walmart."train"
(
    "date"      varchar(12),
    "store_nbr" integer,
    "item_nbr"  integer,
    "units"     integer DEFAULT NULL,
    PRIMARY KEY ("store_nbr", "date", "item_nbr")
);

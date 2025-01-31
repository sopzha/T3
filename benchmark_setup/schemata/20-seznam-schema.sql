CREATE SCHEMA seznam;
DROP TABLE IF EXISTS seznam."client";

CREATE TABLE seznam."client" (
                          "client_id" integer ,
                          "kraj" varchar(255) DEFAULT NULL,
                          "obor" varchar(255) DEFAULT NULL,
                          PRIMARY KEY ("client_id")
) ;

DROP TABLE IF EXISTS seznam."dobito";

CREATE TABLE seznam."dobito" (
                          "client_id" integer DEFAULT NULL,
                          "month_year_datum_transakce" varchar(255) ,
                          "sluzba" varchar(255) ,
                          "kc_dobito" decimal(10,2)
) ;

DROP TABLE IF EXISTS seznam."probehnuto";

CREATE TABLE seznam."probehnuto" (
                              "client_id" integer DEFAULT NULL,
                              "month_year_datum_transakce" varchar(255) ,
                              "sluzba" varchar(255) DEFAULT NULL,
                              "kc_proklikano" decimal(10,2)
) ;

DROP TABLE IF EXISTS seznam."probehnuto_mimo_penezenku";

CREATE TABLE seznam."probehnuto_mimo_penezenku" (
                                             "client_id" integer ,
                                             "Month/Year" varchar(12) ,
                                             "probehla_inzerce_mimo_penezenku" varchar(255) DEFAULT NULL,
                                             PRIMARY KEY ("client_id","Month/Year")
) ;

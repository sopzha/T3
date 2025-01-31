CREATE SCHEMA fhnk;
DROP TABLE IF EXISTS fhnk."pripady";

CREATE TABLE fhnk."pripady" (
                           "Identifikace_pripadu" integer ,
                           "Identifikator_pacienta" integer ,
                           "Kod_zdravotni_pojistovny" integer ,
                           "Datum_prijeti" varchar(255) ,
                           "Datum_propusteni" varchar(255) ,
                           "Delka_hospitalizace" integer ,
                           "Vekovy_Interval_Pacienta" varchar(255) ,
                           "Pohlavi_pacienta" char(1) ,
                           "Zakladni_diagnoza" varchar(255) ,
                           "Seznam_vedlejsich_diagnoz" varchar(255) ,
                           "DRG_skupina" integer ,
                           "PSC" char(5) DEFAULT NULL,
                           PRIMARY KEY ("Identifikace_pripadu")
) ;

DROP TABLE IF EXISTS fhnk."vykony";

CREATE TABLE fhnk."vykony" (
                          "Identifikace_pripadu" integer ,
                          "Datum_provedeni_vykonu" varchar(12) ,
                          "Typ_polozky" integer ,
                          "Kod_polozky" integer ,
                          "Pocet" integer ,
                          "Body" integer ,
                          PRIMARY KEY ("Identifikace_pripadu","Datum_provedeni_vykonu","Kod_polozky")
) ;

DROP TABLE IF EXISTS fhnk."zup";

CREATE TABLE fhnk."zup" (
                       "Identifikace_pripadu" integer ,
                       "Datum_provedeni_vykonu" varchar(12) ,
                       "Typ_polozky" integer DEFAULT NULL,
                       "Kod_polozky" integer ,
                       "Pocet" decimal(10,2) DEFAULT NULL,
                       "Cena" decimal(10,2) DEFAULT NULL,
                       PRIMARY KEY ("Identifikace_pripadu","Datum_provedeni_vykonu","Kod_polozky")
) ;

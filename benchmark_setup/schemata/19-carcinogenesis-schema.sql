CREATE SCHEMA carcinogenesis;
DROP TABLE IF EXISTS carcinogenesis."atom";

CREATE TABLE carcinogenesis."atom" (
                        "atomid" char(13) ,
                        "drug" char(10) DEFAULT NULL,
                        "atomtype" char(100) DEFAULT NULL,
                        "charge" char(100) DEFAULT NULL,
                        "name" char(2) DEFAULT NULL,
                        PRIMARY KEY ("atomid")
) ;

DROP TABLE IF EXISTS carcinogenesis."canc";

CREATE TABLE carcinogenesis."canc" (
                        "drug_id" char(10) ,
                        "class" char(1) DEFAULT NULL,
                        PRIMARY KEY ("drug_id")
) ;

DROP TABLE IF EXISTS carcinogenesis."sbond_1";

CREATE TABLE carcinogenesis."sbond_1" (
                           "id" integer ,
                           "drug" char(10) DEFAULT NULL,
                           "atomid" char(100) DEFAULT NULL,
                           "atomid_2" char(100) DEFAULT NULL,
                           PRIMARY KEY ("id")
) ;

DROP TABLE IF EXISTS carcinogenesis."sbond_2";

CREATE TABLE carcinogenesis."sbond_2" (
                           "id" integer ,
                           "drug" char(10) DEFAULT NULL,
                           "atomid" char(100) DEFAULT NULL,
                           "atomid_2" char(100) DEFAULT NULL,
                           PRIMARY KEY ("id")
) ;

DROP TABLE IF EXISTS carcinogenesis."sbond_3";

CREATE TABLE carcinogenesis."sbond_3" (
                           "id" integer ,
                           "drug" char(8) DEFAULT NULL,
                           "atomid" char(100) DEFAULT NULL,
                           "atomid_2" char(100) DEFAULT NULL,
                           PRIMARY KEY ("id")
) ;

DROP TABLE IF EXISTS carcinogenesis."sbond_7";

CREATE TABLE carcinogenesis."sbond_7" (
                           "id" integer ,
                           "drug" char(9) DEFAULT NULL,
                           "atomid" char(100) DEFAULT NULL,
                           "atomid_2" char(100) DEFAULT NULL,
                           PRIMARY KEY ("id")
) ;

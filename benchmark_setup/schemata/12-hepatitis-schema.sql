CREATE SCHEMA hepatitis;
DROP TABLE IF EXISTS hepatitis."Bio";

CREATE TABLE hepatitis."Bio" (
                       "fibros" varchar(45) ,
                       "activity" varchar(45) ,
                       "b_id" integer ,
                       PRIMARY KEY ("b_id")
) ;

DROP TABLE IF EXISTS hepatitis."dispat";

CREATE TABLE hepatitis."dispat" (
                          "m_id" integer  DEFAULT 0,
                          "sex" varchar(45) DEFAULT NULL,
                          "age" varchar(45) DEFAULT NULL,
                          "Type" varchar(45) DEFAULT NULL,
                          PRIMARY KEY ("m_id")
) ;

DROP TABLE IF EXISTS hepatitis."indis";

CREATE TABLE hepatitis."indis" (
                         "got" varchar(10) DEFAULT NULL,
                         "gpt" varchar(10) DEFAULT NULL,
                         "alb" varchar(45) DEFAULT NULL,
                         "tbil" varchar(45) DEFAULT NULL,
                         "dbil" varchar(45) DEFAULT NULL,
                         "che" varchar(45) DEFAULT NULL,
                         "ttt" varchar(45) DEFAULT NULL,
                         "ztt" varchar(45) DEFAULT NULL,
                         "tcho" varchar(45) DEFAULT NULL,
                         "tp" varchar(45) DEFAULT NULL,
                         "in_id" integer ,
                         PRIMARY KEY ("in_id")
) ;

DROP TABLE IF EXISTS hepatitis."inf";

CREATE TABLE hepatitis."inf" (
                       "dur" varchar(45) DEFAULT NULL,
                       "a_id" integer  DEFAULT 0,
                       PRIMARY KEY ("a_id")
) ;

DROP TABLE IF EXISTS hepatitis."rel11";

CREATE TABLE hepatitis."rel11" (
                         "b_id" integer  DEFAULT 0,
                         "m_id" integer  DEFAULT 0,
                         PRIMARY KEY ("b_id","m_id")
) ;

DROP TABLE IF EXISTS hepatitis."rel12";

CREATE TABLE hepatitis."rel12" (
                         "in_id" integer  DEFAULT 0,
                         "m_id" integer  DEFAULT 0,
                         PRIMARY KEY ("in_id","m_id")
) ;

DROP TABLE IF EXISTS hepatitis."rel13";

CREATE TABLE hepatitis."rel13" (
                         "a_id" integer  DEFAULT 0,
                         "m_id" integer  DEFAULT 0,
                         PRIMARY KEY ("a_id","m_id")
) ;


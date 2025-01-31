CREATE SCHEMA movielens;

DROP TABLE IF EXISTS movielens."actors";

CREATE TABLE movielens."actors" (
                          "actorid" integer ,
                          "a_gender" varchar(255) ,
                          "a_quality" integer ,
                          PRIMARY KEY ("actorid")
) ;

DROP TABLE IF EXISTS movielens."directors";

CREATE TABLE movielens."directors" (
                             "directorid" integer ,
                             "d_quality" integer ,
                             "avg_revenue" integer ,
                             PRIMARY KEY ("directorid")
) ;

DROP TABLE IF EXISTS movielens."movies";

CREATE TABLE movielens."movies" (
                          "movieid" integer  DEFAULT 0,
                          "year" integer ,
                          "isEnglish" varchar(255) ,
                          "country" varchar(50) ,
                          "runningtime" integer ,
                          PRIMARY KEY ("movieid")
) ;

DROP TABLE IF EXISTS movielens."movies2actors";

CREATE TABLE movielens."movies2actors" (
                                 "movieid" integer ,
                                 "actorid" integer ,
                                 "cast_num" integer ,
                                 PRIMARY KEY ("movieid","actorid")
) ;

DROP TABLE IF EXISTS movielens."movies2directors";

CREATE TABLE movielens."movies2directors" (
                                    "movieid" integer ,
                                    "directorid" integer ,
                                    "genre" varchar(15) ,
                                    PRIMARY KEY ("movieid","directorid")
) ;

DROP TABLE IF EXISTS movielens."u2base";

CREATE TABLE movielens."u2base" (
                          "userid" integer  DEFAULT 0,
                          "movieid" integer ,
                          "rating" varchar(45) ,
                          PRIMARY KEY ("userid","movieid")
) ;

DROP TABLE IF EXISTS movielens."users";

CREATE TABLE movielens."users" (
                         "userid" integer  DEFAULT 0,
                         "age" varchar(5) ,
                         "u_gender" varchar(5) ,
                         "occupation" varchar(45) ,
                         PRIMARY KEY ("userid")
) ;


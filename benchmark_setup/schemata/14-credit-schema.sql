CREATE SCHEMA credit;
DROP TABLE IF EXISTS credit."category";

CREATE TABLE credit."category" (
                            "category_no" integer ,
                            "category_desc" varchar(31) ,
                            "category_code" char(2) ,
                            PRIMARY KEY ("category_no")
) ;

DROP TABLE IF EXISTS credit."charge";

CREATE TABLE credit."charge" (
                          "charge_no" integer ,
                          "member_no" integer ,
                          "provider_no" integer ,
                          "category_no" integer ,
                          "charge_dt" varchar(255) ,
                          "charge_amt" decimal(19,4) ,
                          "statement_no" integer ,
                          "charge_code" char(2) ,
                          PRIMARY KEY ("charge_no")
) ;

DROP TABLE IF EXISTS credit."corporation";

CREATE TABLE credit."corporation" (
                               "corp_no" integer ,
                               "corp_name" varchar(31) ,
                               "street" varchar(15) ,
                               "city" varchar(15) ,
                               "state_prov" char(2) ,
                               "country" char(2) ,
                               "mail_code" char(10) ,
                               "phone_no" char(13) ,
                               "expr_dt" varchar(255) ,
                               "region_no" integer ,
                               "corp_code" char(2) ,
                               PRIMARY KEY ("corp_no")
) ;

DROP TABLE IF EXISTS credit."member";

CREATE TABLE credit."member" (
                          "member_no" integer ,
                          "lastname" varchar(15) ,
                          "firstname" varchar(15) ,
                          "middleinitial" char(1) DEFAULT NULL,
                          "street" varchar(15) ,
                          "city" varchar(15) ,
                          "state_prov" char(2) ,
                          "country" char(2) ,
                          "mail_code" char(10) ,
                          "phone_no" char(13) DEFAULT NULL,
                          "photograph" bytea DEFAULT NULL,
                          "issue_dt" varchar(255) ,
                          "expr_dt" varchar(255) ,
                          "region_no" integer ,
                          "corp_no" integer DEFAULT NULL,
                          "prev_balance" decimal(19,4) DEFAULT NULL,
                          "curr_balance" decimal(19,4) DEFAULT NULL,
                          "member_code" char(2) ,
                          PRIMARY KEY ("member_no")
) ;

DROP TABLE IF EXISTS credit."payment";

CREATE TABLE credit."payment" (
                           "payment_no" integer ,
                           "member_no" integer ,
                           "payment_dt" varchar(255) ,
                           "payment_amt" decimal(19,4) ,
                           "statement_no" integer DEFAULT NULL,
                           "payment_code" char(2) ,
                           PRIMARY KEY ("payment_no")
) ;

DROP TABLE IF EXISTS credit."provider";

CREATE TABLE credit."provider" (
                            "provider_no" integer ,
                            "provider_name" varchar(15) ,
                            "street" varchar(15) ,
                            "city" varchar(15) ,
                            "state_prov" char(2) ,
                            "mail_code" char(10) ,
                            "country" char(2) ,
                            "phone_no" char(13) ,
                            "issue_dt" varchar(255) ,
                            "expr_dt" varchar(255) ,
                            "region_no" integer ,
                            "provider_code" char(2) ,
                            PRIMARY KEY ("provider_no")
) ;

DROP TABLE IF EXISTS credit."region";

CREATE TABLE credit."region" (
                          "region_no" integer ,
                          "region_name" varchar(15) ,
                          "street" varchar(15) ,
                          "city" varchar(15) ,
                          "state_prov" char(2) ,
                          "country" char(2) ,
                          "mail_code" char(10) ,
                          "phone_no" char(13) ,
                          "region_code" char(2) ,
                          PRIMARY KEY ("region_no")
) ;

DROP TABLE IF EXISTS credit."statement";

CREATE TABLE credit."statement" (
                             "statement_no" integer ,
                             "member_no" integer ,
                             "statement_dt" varchar(255) ,
                             "due_dt" varchar(255) ,
                             "statement_amt" decimal(19,4) ,
                             "statement_code" char(2) ,
                             PRIMARY KEY ("statement_no")
) ;

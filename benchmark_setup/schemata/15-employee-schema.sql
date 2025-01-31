CREATE SCHEMA employee;
DROP TABLE IF EXISTS employee."departments";

CREATE TABLE employee."departments" (
                               "dept_no" char(6) ,
                               "dept_name" varchar(40) ,
                               PRIMARY KEY ("dept_no")
) ;

DROP TABLE IF EXISTS employee."dept_emp";

CREATE TABLE employee."dept_emp" (
                            "emp_no" integer ,
                            "dept_no" char(6) ,
                            "from_date" varchar(255) ,
                            "to_date" varchar(255) ,
                            PRIMARY KEY ("emp_no","dept_no")
) ;

DROP TABLE IF EXISTS employee."dept_manager";

CREATE TABLE employee."dept_manager" (
                                "dept_no" char(6) ,
                                "emp_no" integer ,
                                "from_date" varchar(255) ,
                                "to_date" varchar(255) ,
                                PRIMARY KEY ("emp_no","dept_no")
) ;

DROP TABLE IF EXISTS employee."employees";

CREATE TABLE employee."employees" (
                             "emp_no" integer ,
                             "birth_date" varchar(255) ,
                             "first_name" varchar(14) ,
                             "last_name" varchar(16) ,
                             "gender" varchar(255) ,
                             "hire_date" varchar(255) ,
                             PRIMARY KEY ("emp_no")
) ;

DROP TABLE IF EXISTS employee."salaries";

CREATE TABLE employee."salaries" (
                            "emp_no" integer ,
                            "salary" integer ,
                            "from_date" varchar(12) ,
                            "to_date" varchar(255) ,
                            PRIMARY KEY ("emp_no","from_date")
) ;

DROP TABLE IF EXISTS employee."titles";

CREATE TABLE employee."titles" (
                          "emp_no" integer ,
                          "title" varchar(20) ,
                          "from_date" varchar(12) ,
                          "to_date" varchar(255) DEFAULT NULL,
                          PRIMARY KEY ("emp_no","title","from_date")
) ;

CREATE SCHEMA ssb;
drop table if exists ssb.customer;
CREATE TABLE ssb.customer 
(
    c_custkey    integer     not null,
    c_name       varchar(25) not null,
    c_address    varchar(25) not null,
    c_city       varchar(10) not null,
    c_nation     varchar(15) not null,
    c_region     varchar(12) not null,
    c_phone      varchar(15) not null,
    c_mktsegment varchar(10) not null,
    primary key (c_custkey)
);

drop table if exists ssb.part;
CREATE TABLE ssb.part 
(
    p_partkey   integer     not null,
    p_name      varchar(22) not null,
    p_mfgr      varchar(6)  not null,
    p_category  varchar(7)  not null,
    p_brand1    varchar(9)  not null,
    p_color     varchar(11) not null,
    p_type      varchar(25) not null,
    p_size      integer     not null,
    p_container varchar(10) not null,
    primary key (p_partkey)
);

drop table if exists ssb.supplier;
CREATE TABLE ssb.supplier 
(
    s_suppkey integer     not null,
    s_name    varchar(25) not null,
    s_address varchar(25) not null,
    s_city    varchar(10) not null,
    s_nation  varchar(15) not null,
    s_region  varchar(12) not null,
    s_phone   varchar(15) not null,
    primary key (s_suppkey)
);

drop table if exists ssb.lineorder;
CREATE TABLE ssb.lineorder 
(
    lo_orderkey        BIGINT      not null,
    lo_linenumber      BIGINT      not null,
    lo_custkey         integer     not null,
    lo_partkey         integer     not null,
    lo_suppkey         integer     not null,
    lo_orderdate       integer     not null,
    lo_orderpriority   varchar(15) not null,
    lo_shippriority    integer     not null,
    lo_quantity        integer     not null,
    lo_extendedprice   integer     not null,
    lo_ordertotalprice integer     not null,
    lo_discount        integer     not null,
    lo_revenue         integer     not null,
    lo_supplycost      integer     not null,
    lo_tax             integer     not null,
    lo_commitdate      integer     not null,
    lo_shipmode        varchar(10) not null
);

drop table if exists ssb.dim_date;
CREATE TABLE ssb.dim_date (
                          d_datekey          integer     not null,
                          d_date             varchar(18) not null,
                          d_dayofweek        varchar(9)  not null,
                          d_month            varchar(9)  not null,
                          d_year             integer     not null,
                          d_yearmonthnum     integer     not null,
                          d_yearmonth        varchar(7)  not null,
                          d_daynuminweek     integer     not null,
                          d_daynuminmonth    integer     not null,
                          d_daynuminyear     integer     not null,
                          d_monthnuminyear   integer     not null,
                          d_weeknuminyear    integer     not null,
                          d_sellingseason    varchar(12) not null,
                          d_lastdayinweekfl  integer     not null,
                          d_lastdayinmonthfl integer     not null,
                          d_holidayfl        integer     not null,
                          d_weekdayfl        integer     not null,
                          primary key (d_datekey)
);

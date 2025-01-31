-- Create the 'airline' schema
CREATE SCHEMA airline;

-- Create tables within the 'airline' schema
DROP TABLE IF EXISTS airline."L_AIRLINE_ID";

CREATE TABLE airline."L_AIRLINE_ID"
(
    "Code"        integer NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_AIRPORT";

CREATE TABLE airline."L_AIRPORT"
(
    "Code"        char(3) NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_AIRPORT_ID";

CREATE TABLE airline."L_AIRPORT_ID"
(
    "Code"        integer NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_AIRPORT_SEQ_ID";

CREATE TABLE airline."L_AIRPORT_SEQ_ID"
(
    "Code"        integer NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_CANCELLATION";

CREATE TABLE airline."L_CANCELLATION"
(
    "Code"        char(1) NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_CITY_MARKET_ID";

CREATE TABLE airline."L_CITY_MARKET_ID"
(
    "Code"        integer NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_DEPARRBLK";

CREATE TABLE airline."L_DEPARRBLK"
(
    "Code"        char(9) NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_DISTANCE_GROUP_250";

CREATE TABLE airline."L_DISTANCE_GROUP_250"
(
    "Code"        integer NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_DIVERSIONS";

CREATE TABLE airline."L_DIVERSIONS"
(
    "Code"        integer NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_MONTHS";

CREATE TABLE airline."L_MONTHS"
(
    "Code"        integer NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_ONTIME_DELAY_GROUPS";

CREATE TABLE airline."L_ONTIME_DELAY_GROUPS"
(
    "Code"        integer NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_QUARTERS";

CREATE TABLE airline."L_QUARTERS"
(
    "Code"        integer NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_STATE_ABR_AVIATION";

CREATE TABLE airline."L_STATE_ABR_AVIATION"
(
    "Code"        char(2) NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_STATE_FIPS";

CREATE TABLE airline."L_STATE_FIPS"
(
    "Code"        integer NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_UNIQUE_CARRIERS";

CREATE TABLE airline."L_UNIQUE_CARRIERS"
(
    "Code"        varchar(255) NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_WEEKDAYS";

CREATE TABLE airline."L_WEEKDAYS"
(
    "Code"        integer NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_WORLD_AREA_CODES";

CREATE TABLE airline."L_WORLD_AREA_CODES"
(
    "Code"        integer NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."L_YESNO_RESP";

CREATE TABLE airline."L_YESNO_RESP"
(
    "Code"        integer NOT NULL,
    "Description" varchar(255) DEFAULT NULL,
    PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS airline."On_Time_On_Time_Performance_2016_1";

CREATE TABLE airline."On_Time_On_Time_Performance_2016_1"
(
    "Year"                 integer        DEFAULT NULL,
    "Quarter"              integer        DEFAULT NULL,
    "Month"                integer        DEFAULT NULL,
    "DayofMonth"           integer        DEFAULT NULL,
    "DayOfWeek"            integer        DEFAULT NULL,
    "FlightDate"           date           DEFAULT NULL,
    "UniqueCarrier"        varchar(255)   DEFAULT 'NULL',
    "AirlineID"            integer NOT NULL,
    "Carrier"              char(2)        DEFAULT NULL,
    "TailNum"              varchar(6)     DEFAULT NULL,
    "FlightNum"            integer        DEFAULT NULL,
    "OriginAirportID"      integer        DEFAULT NULL,
    "OriginAirportSeqID"   integer        DEFAULT NULL,
    "OriginCityMarketID"   integer        DEFAULT NULL,
    "Origin"               char(3)        DEFAULT NULL,
    "OriginCityName"       varchar(34)    DEFAULT NULL,
    "OriginState"          char(2)        DEFAULT NULL,
    "OriginStateFips"      integer        DEFAULT NULL,
    "OriginStateName"      varchar(46)    DEFAULT NULL,
    "OriginWac"            integer        DEFAULT NULL,
    "DestAirportID"        integer        DEFAULT NULL,
    "DestAirportSeqID"     integer        DEFAULT NULL,
    "DestCityMarketID"     integer        DEFAULT NULL,
    "Dest"                 char(3)        DEFAULT NULL,
    "DestCityName"         varchar(34)    DEFAULT NULL,
    "DestState"            char(2)        DEFAULT NULL,
    "DestStateFips"        integer        DEFAULT NULL,
    "DestStateName"        varchar(46)    DEFAULT NULL,
    "DestWac"              integer        DEFAULT NULL,
    "CRSDepTime"           integer        DEFAULT NULL,
    "DepTime"              integer        DEFAULT NULL,
    "DepDelay"             decimal(38, 2) DEFAULT NULL,
    "DepDelayMinutes"      float          DEFAULT NULL,
    "DepDel15"             integer        DEFAULT NULL,
    "DepartureDelayGroups" integer        DEFAULT NULL,
    "DepTimeBlk"           char(9)        DEFAULT NULL,
    "TaxiOut"              float          DEFAULT NULL,
    "WheelsOff"            integer        DEFAULT NULL,
    "WheelsOn"             integer        DEFAULT NULL,
    "TaxiIn"               float          DEFAULT NULL,
    "CRSArrTime"           integer        DEFAULT NULL,
    "ArrTime"              integer        DEFAULT NULL,
    "ArrDelay"             decimal(38, 2) DEFAULT NULL,
    "ArrDelayMinutes"      float          DEFAULT NULL,
    "ArrDel15"             integer        DEFAULT NULL,
    "ArrivalDelayGroups"   integer        DEFAULT NULL,
    "ArrTimeBlk"           char(9)        DEFAULT NULL,
    "Cancelled"            integer        DEFAULT NULL,
    "CancellationCode"     char(1)        DEFAULT NULL,
    "Diverted"             integer        DEFAULT NULL,
    "CRSElapsedTime"       float          DEFAULT NULL,
    "ActualElapsedTime"    float          DEFAULT NULL,
    "AirTime"              float          DEFAULT NULL,
    "Flights"              float          DEFAULT NULL,
    "Distance"             float          DEFAULT NULL,
    "DistanceGroup"        integer        DEFAULT NULL,
    "CarrierDelay"         decimal(38, 2) DEFAULT NULL,
    "WeatherDelay"         decimal(38, 2) DEFAULT NULL,
    "NASDelay"             decimal(38, 2) DEFAULT NULL,
    "SecurityDelay"        decimal(38, 2) DEFAULT NULL,
    "LateAircraftDelay"    decimal(38, 2) DEFAULT NULL,
    "FirstDepTime"         decimal(38, 2) DEFAULT NULL,
    "TotalAddGTime"        decimal(38, 2) DEFAULT NULL,
    "LongestAddGTime"      decimal(38, 2) DEFAULT NULL,
    "DivAirportLandings"   integer        DEFAULT NULL,
    "DivReachedDest"       decimal(38, 2) DEFAULT NULL,
    "DivActualElapsedTime" decimal(38, 2) DEFAULT NULL,
    "DivArrDelay"          decimal(38, 2) DEFAULT NULL,
    "DivDistance"          decimal(38, 2) DEFAULT NULL,
    "Div1Airport"          char(3)        DEFAULT NULL,
    "Div1AirportID"        integer        DEFAULT NULL,
    "Div1AirportSeqID"     integer        DEFAULT NULL,
    "Div1WheelsOn"         decimal(38, 2) DEFAULT NULL,
    "Div1TotalGTime"       decimal(38, 2) DEFAULT NULL,
    "Div1LongestGTime"     decimal(38, 2) DEFAULT NULL,
    "Div1WheelsOff"        decimal(38, 2) DEFAULT NULL,
    "Div1TailNum"          varchar(6)     DEFAULT NULL,
    "Div2Airport"          char(3)        DEFAULT NULL,
    "Div2AirportID"        integer        DEFAULT NULL,
    "Div2AirportSeqID"     integer        DEFAULT NULL,
    "Div2WheelsOn"         decimal(38, 2) DEFAULT NULL,
    "Div2TotalGTime"       decimal(38, 2) DEFAULT NULL,
    "Div2LongestGTime"     decimal(38, 2) DEFAULT NULL
);

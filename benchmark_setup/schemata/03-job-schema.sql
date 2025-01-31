create schema job;

CREATE TABLE job.aka_name (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    name text NOT NULL,
    imdb_index character varying(12),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE job.aka_title (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    title text NOT NULL,
    imdb_index character varying(12),
    kind_id integer NOT NULL,
    production_year integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note text,
    md5sum character varying(32)
);

CREATE TABLE job.cast_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    movie_id integer NOT NULL,
    person_role_id integer,
    note text,
    nr_order integer,
    role_id integer NOT NULL
);

CREATE TABLE job.char_name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL,
    imdb_index character varying(12),
    imdb_id integer,
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE job.comp_cast_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(32) NOT NULL
);

CREATE TABLE job.company_name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL,
    country_code character varying(255),
    imdb_id integer,
    name_pcode_nf character varying(5),
    name_pcode_sf character varying(5),
    md5sum character varying(32)
);

CREATE TABLE job.company_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(32) NOT NULL
);

CREATE TABLE job.complete_cast (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer,
    subject_id integer NOT NULL,
    status_id integer NOT NULL
);

CREATE TABLE job.info_type (
    id integer NOT NULL PRIMARY KEY,
    info character varying(32) NOT NULL
);

CREATE TABLE job.keyword (
    id integer NOT NULL PRIMARY KEY,
    keyword text NOT NULL,
    phonetic_code character varying(5)
);

CREATE TABLE job.kind_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(15) NOT NULL
);

CREATE TABLE job.link_type (
    id integer NOT NULL PRIMARY KEY,
    link character varying(32) NOT NULL
);

CREATE TABLE job.movie_companies (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    company_id integer NOT NULL,
    company_type_id integer NOT NULL,
    note text
);

CREATE TABLE job.movie_info (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text
);

CREATE TABLE job.movie_info_idx (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text
);

CREATE TABLE job.movie_keyword (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    keyword_id integer NOT NULL
);

CREATE TABLE job.movie_link (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    linked_movie_id integer NOT NULL,
    link_type_id integer NOT NULL
);

CREATE TABLE job.name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL,
    imdb_index character varying(12),
    imdb_id integer,
    gender character varying(1),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE job.person_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text
);

CREATE TABLE job.role_type (
    id integer NOT NULL PRIMARY KEY,
    role character varying(32) NOT NULL
);

CREATE TABLE job.title (
    id integer NOT NULL PRIMARY KEY,
    title text NOT NULL,
    imdb_index character varying(12),
    kind_id integer NOT NULL,
    production_year integer,
    imdb_id integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years character varying(49),
    md5sum character varying(32)
);

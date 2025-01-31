COPY basketball."awards_coaches" FROM '/data/basketball/awards_coaches.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY basketball."awards_players" FROM '/data/basketball/awards_players.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY basketball."coaches" FROM '/data/basketball/coaches.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY basketball."draft" FROM '/data/basketball/draft.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY basketball."player_allstar" FROM '/data/basketball/player_allstar.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY basketball."players" FROM '/data/basketball/players.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY basketball."players_teams" FROM '/data/basketball/players_teams.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY basketball."series_post" FROM '/data/basketball/series_post.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;
COPY basketball."teams" FROM '/data/basketball/teams.csv' DELIMITER '	' QUOTE '"' ESCAPE '\' NULL 'NULL' CSV HEADER;


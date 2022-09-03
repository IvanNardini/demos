-- SET GLOBAL local_infile = 'ON';

CREATE TABLE CUSTOMERID (
    id INT NOT NULL AUTO_INCREMENT, 
    customerId varchar(255),
    PRIMARY KEY (id)
);

CREATE TABLE TRANSACTIONS (
    id INT NOT NULL AUTO_INCREMENT,
    customerId varchar(255),
    products varchar(255),
    PRIMARY KEY (id)
);

LOAD DATA LOCAL INFILE '/var/lib/mysql-files/recommend_1.csv' into table CUSTOMERID
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n' 
IGNORE 1 LINES;

LOAD DATA LOCAL INFILE  '/var/lib/mysql-files/trx_data.csv' into table TRANSACTIONS
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n' 
IGNORE 1 LINES;
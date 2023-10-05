# NOTE name need to be Dockerfile
FROM mysql:latest
COPY ./databse_train.sql /docker-entrypoint-initdb.d/
COPY ./databse_test.sql /docker-entrypoint-initdb.d/
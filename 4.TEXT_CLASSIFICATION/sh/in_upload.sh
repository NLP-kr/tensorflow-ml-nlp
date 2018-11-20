#!/bin/bash

cd ..

rsync -avz --exclude=".git" --exclude="data_out/" --delete ./ companyai8way@10.64.50.108:/home/companyai8way/ryan/classifier

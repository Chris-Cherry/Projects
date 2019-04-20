#!/usr/bin/env bash

# Start by resetting IP address
killall -HUP tor

if [ ! -f htmls.csv ]
then
    sudo -u bigboy python3 getHtmls.py
fi

if [ ! -f fin.csv ]
then
    killall -HUP tor
    sudo -u bigboy python3 getTickers.py
fi

# Check how many lines in fin and html as condition for running getTickers.py
fin_lines=$(wc -l fin.csv | cut -f1 -d' ')
html_lines=$(wc -l htmls.csv | cut -f1 -d' ')

while [ $fin_lines -lt $html_lines ]
do
    killall -HUP tor
    echo 'Running because $fin_lines less than $html_lines'
    sudo -u bigboy python3 getTickers.py
    fin_lines=$(wc -l fin.csv | cut -f1 -d' ')
    html_lines=$(wc -l htmls.csv | cut -f1 -d' ')
done

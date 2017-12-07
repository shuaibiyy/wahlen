# Wahlen

Learning python by exploring Germany's 2017 election results.

## Requirements

* [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)

## Usage

The python script produces the output of some of the steps involved in the allocation of seats in the Bundestag.
Available output:

__Second votes (Zweitstimmen)__:
```
python wahl.py second_votes # generates _second_votes.csv

python wahl.py second_votes_chart # opens up a bar chart in a new window
```

__Direct seats and list seats (Direktmandat, Listenmandat and Überhang)__:
```
python wahl.py direct_list_seats # generates _direct_list_seats.csv
```

__Final Bundestag Seat Distribution__:
```
python wahl.py bundestag_seats # generates _bundestag_seats.csv
```

## Tests

Tests are written using [doctest](https://docs.python.org/3/library/doctest.html).

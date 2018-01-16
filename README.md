# Wahlen

Learning python by exploring Germany's 2017 election results.

## Usage

The python script produces the output of some of the steps involved in the allocation of seats in the Bundestag.
Available output:

__Second votes (Zweitstimmen)__:
```
python wahl.py second_votes # generates _second_votes.csv

python wahl.py second_votes_chart # opens up a bar chart in a new window
```

__Display a chart of second vote percentages across parties in Germany__:
```
python wahl.py second_votes_chart
```

__Direct seats and list seats (Direktmandat, Listenmandat and Überhang)__:
```
python wahl.py direct_list_seats # generates _direct_list_seats.csv
```

__Bundestag Seat Distribution__:
```
python wahl.py bundestag_seats # generates _bundestag_seats.csv
```

__Candidates elected to the Bundestag__:

_This takes about 1-2 minutes to complete on my macbook pro 2016 as it crawls the [election site](https://www.bundeswahlleiter.de) to extract the elected candidates._
```
python wahl.py elected_candidates # generates _elected_candidates.csv
```

__Display a map of directly elected candidates to the Bundestag__:

_This takes about 1-2 minutes to complete on my macbook pro 2016 as it crawls the [election site](https://www.bundeswahlleiter.de) to extract the elected candidates._
```
python wahl.py directly_elected_candidates_map
```

## Tests

Tests are written using [doctest](https://docs.python.org/3/library/doctest.html).

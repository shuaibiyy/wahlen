# coding=utf-8
# Written using python 2.7.10
import csv
import pprint


def fetch_rows(csv_path):
    """Open a CSV file and read its lines."""
    rows = []

    with open(csv_path, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    return rows


def get_csv_values():
    """Return the semi-colon separated values in the csv as a list."""
    rows = fetch_rows('./ergebnisse.csv')

    # Remove column headings
    del rows[0]

    return map(lambda x: x[0].split(';'), rows)


def parties():
    """Return a set of all parties in the csv."""
    vs = get_csv_values()

    # party names are at the 3rd index.
    initial = list(set(map(lambda x: x[2], vs)))

    not_parties = ['Wahlberechtigte', 'Wähler', 'Ungültige', 'Gültige']

    return filter(lambda x: x not in not_parties, initial)


def second_vote(party, value):
    """Return a zweitstimmen if it matches a party."""
    if value[2] == party:
        # zweitstimmen is at index #4.
        return value[4]

    return '-'


def second_votes(party, values):
    """Return all zweitstimmen for a party."""
    return map(lambda x: second_vote(party, x), values)


def filter_dashes((k, v)):
    """Remove dashes from seconds in a 2-tuple."""
    return k, filter(lambda x: x != '-', v)


def aggregate((k, v)):
    """Aggregate values of all seconds in a 2-tuple."""
    return k, reduce(lambda acc, x: acc + int(x), v, 0)


def cleanse_votes(dirty):
    """Return a sorted list of parties with non-zero votes."""
    unsorted_votes = map(aggregate, map(filter_dashes, dirty))

    non_zero_votes = filter(lambda x: x[1] != 0, unsorted_votes)

    return sorted(non_zero_votes, key=lambda tup: tup[1], reverse=True)


def percentage(num, denom):
    """Return the percentage string of a numerator to its denominator."""
    return '{:.4f}'.format((float(num) / denom) * 100)


def votes_with_percentages(vote_total, all_votes):
    """Return a list of parties along with their votes and percentage share."""
    return map(lambda (k, v): (k, percentage(v, vote_total)), all_votes)


def votes():
    """Compute zweitstimmen for all parties in the csv."""
    vs = get_csv_values()
    ps = parties()

    # contains `-` values for missing votes.
    unfiltered_values = map(lambda x: (x, second_votes(x, vs)), ps)

    cleansed_votes = cleanse_votes(unfiltered_values)

    vote_total = reduce(lambda acc, (k, v): acc + v, cleansed_votes, 0)

    return votes_with_percentages(vote_total, cleansed_votes)


print('Party;Percentage')
for row in votes():
    print('{0};{1}'.format(row[0], row[1]))


# pprint.pprint(votes())

"""
Python 3.6

This script performs the following:

    - Fetch organism proteome FASTA file containing all protein sequences
    - Clean FASTA data
    - Scrape uniprot data based to retrieve protein subcellular localization
    - Write uniprot ID, protein sequence, and subcellular localization to csv

Takes 8 min to run for ~5000 proteins proteome on old a** macbookair with FIOS
connection

Libraries:
    - Biopython
    - Pandas

by MAS 2019
"""

import urllib.request
import gzip
import re
from Bio import SeqIO
import pandas as pd
import time

###############################################################################
# Enter information below

# Name of Proteome
NAME = "ecoli_proteome"

# URL to compressed (gz) file with all sequences
URL = "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/" \
      "knowledgebase/reference_proteomes/Bacteria/UP000000625_83333.fasta.gz"


###############################################################################

# Timer
start_time = time.time()


def try_url(url, name):
    """
    Check URL, fetch, and extract FASTA file from compressed gz
    :param url: URL to uniprot proteome FASTA file
    :param name: str, proteome name
    :return: fasta file as list
    """
    try:
        urllib.request.urlretrieve(url, "./" + name + ".gz")
        return extract_file(name)
    except urllib.error.URLError:
        print("URL not valid. Please check and try again")


def extract_file(name):
    """
    Extract FASTA file from compressed gz
    :param name: str, proteome name
    :return: list, extracted FASTA file split by protein
    """
    f = gzip.open("./" + name + ".gz", "rt")
    file_contents = f.read()
    f.close()
    parsed_file = file_contents.split(">")
    # Remove Empty Rows
    parsed_file = [entry for entry in parsed_file if len(entry) != 0]
    return parsed_file


def add_to_dataframe(entry):
    """
    Extract ID and sequence from each FASTA string (use regex).
    :param entry: str, FASTA string
    :return: ID, seq (both str)
    """
    # Get sequence
    seq_index = re.search("SV=", entry)
    seq = entry[(seq_index.end() + 1):]
    # Get rid of line breaks
    seq = ''.join(seq.split("\n"))
    # Get ID
    code = re.search("(?<=\|).*(?=\|)", entry)
    return code.group(), seq


def get_local(entry):
    """
    Scrape uniprot database for the subcellular localization of each protein
    Define "undocumented" if no subcellular localization is reported
    :param entry: str, uniprot ID extracted from FASTA file
    :return: str, localization in {undocumented, soluble, membrane}
    """
    handle = urllib.request.urlopen(
        "https://www.uniprot.org/uniprot/" + entry + ".xml")
    record = SeqIO.read(handle, "uniprot-xml")
    try:
        local = record.annotations['comment_subcellularlocation_location'][0]
        if "membrane" in local.lower().split():
            local = "membrane"
        elif local == "Cytoplasm" or local == "Periplasm":
            local = "soluble"
    except KeyError:
        local = "undocumented"
    return local


def generate_csv(url=URL, name=NAME):
    """
    Run everything, convert uniprot FASTA format to tidy pandas dataframe
    containing only uniprot ID, sequence, and localization
    :param url: str, URL to uniprot proteome FASTA file
    :param name: str, proteome name
    :return: None
    """
    proteome = try_url(url, name)
    df = pd.DataFrame(columns=["ID", "Sequence", "Local"])

    for index, sequence in enumerate(proteome):
        protid, aa = add_to_dataframe(sequence)
        df.loc[index] = [protid, aa, get_local(protid)]

    df.to_csv("./ecoli_proteome.csv", index=False)


if __name__ == '__main__':
    generate_csv()
    print('run time = ' + str(time.time() - start_time) + ' s')

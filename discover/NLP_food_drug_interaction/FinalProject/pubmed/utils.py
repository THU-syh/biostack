import sys
import re
import json
import requests
from xml.etree import ElementTree


class PubMedQuery:

    # number of instances
    COUNT = 0

    def __init__(self, search_term, max_results=500):
        '''
        Parameters:
        -----------
        search term: a string. More information on
                     https://www.ncbi.nlm.nih.gov/books/NBK3827/#pubmedhelp.Advanced_Search
                     For example: 'ACE inhibitor'
        max_results: numeric.
        '''

        self.max_results = max_results
        self.start = self.COUNT * int(max_results)
        # look up search term in pubmed database
        search_url = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed'
        self.res_search = requests.get(url=search_url,
                                       params={"term": search_term, "retmode": 'json', "retmax": self.max_results,
                                               "retstart": self.start})
        # raise error if something went wrong
        self.res_search.raise_for_status()
        # increase instance count in order to keep track of the starting point in the result list
        PubMedQuery.COUNT += 1

    @property
    def n_articles(self):
        return self.res_search.json()['esearchresult']["count"]

    def id_getter(self):
        '''Returns pubmed ids for a search term.'''

        # extract ids from json result page
        res_search_ids = self.res_search.json()['esearchresult']['idlist']
        # turn id list to id string
        res_ids = ','.join(res_search_ids)
        return res_ids

    @staticmethod
    def abstract_getter(ids):
        '''Returns abstracts for pubmed ids.

        Parameters:
        -----------
        ids: a string of pubmed ids separated by commas.

        Returns:
        ---------
        A dictionary of ids as keys and abstracts as values.
        '''

        # store abstracts in dictionary (key = id, values: = abstract)
        abstract_dict = {}
        # download abstracts that match the ids
        abstract_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&rettype=abstract'
        res = requests.get(url = abstract_url, params = {"id": ids})
        # check if everything went well (a usual error is if we asked for too many ids)
        res.raise_for_status()
        # result is an XML that needs to be parsed
        xml_res = ElementTree.fromstring(res.content)
        # save and return dictionary 
        for i, abst in enumerate(xml_res.iter('AbstractText')):
            abstract_dict[i] = abst.text
        return abstract_dict


def download_all_abstracts(search_term, max_results):
    '''Downloads all of the PubMed abstracts corresponding to search_term and saves it to json files
       contaning a maximum of max_results abstracts'''
    
    more_abstracts = True
    while more_abstracts is True:
        # start pubmed query to download a maximum of max_results abstracts
        query = PubMedQuery(search_term, max_results)
        ids = query.id_getter()
        abstracts = query.abstract_getter(ids)
        # check stopping condition
        TOTAL = query.n_articles
        STARTNUM = int(max_results)*PubMedQuery.COUNT
        more_abstracts = (STARTNUM + int(max_results)) < int(query.n_articles)
        # write results to jsons
        json_file = 'pbabstract' + str(PubMedQuery.COUNT) + '.json'
        print 'Saving to ' + json_file
        print str(STARTNUM) + '/' + TOTAL + ' downloaded'
        with open(json_file, 'w') as outfile:
            json.dump(abstracts, outfile, indent=4)



def ace_substitutor(text, token = 'ACEI'):
    '''Substitutes token in place of different spellings of ACE inhibitor'''
    
    ace_options = r'^(\bangiotensin converting enzyme (ace) inhibitor+?)(s\b|\b)|(\bangiotensin-converting enzyme (ace) inhibitor+?)(s\b|\b)|(\bace inhibitor+?)(s\b|\b)|(\bace\b)|(\bacei\b)|(\bangiotensin-converting enzyme inhibitor+?)(s\b|\b) \(acei+?(s\b|\b)\)|(\bangiotensin converting enzyme inhibitor+?)(s\b|\b) \(acei+?(s\b|\b)\)|(\bangiotensin-converting enzyme inhibitor+?)(s\b|\b)|(\bangiotensin-converting enzyme+?)(s\b|\b)|(\bangiotensin converting enzyme inhibitor+?)(s\b|\b)|(\bangiotensin converting enzyme+?)(s\b|\b)'
    return re.sub(ace_options,'ACEI', text.lower())



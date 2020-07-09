__author__ = 'aub3'
import logging,os,gzip
from collections import defaultdict
import formats
FILES = [
    {'state':'CA','file_type':'CORE','dataset_type':'SID','year':2006,'filename':'CA_SID_2006_CORE.asc.gz','link':'CA_2006_daystoevent.csv.gz'},
    {'state':'CA','file_type':'CORE','dataset_type':'SID','year':2007,'filename':'CA_SID_2007_CORE.asc.gz','link':'CA_2007_daystoevent.csv.gz'},
    {'state':'CA','file_type':'CORE','dataset_type':'SID','year':2008,'filename':'CA_SID_2008_CORE.asc.gz','link':'CA_2008_daystoevent.csv.gz'},
    {'state':'CA','file_type':'CORE','dataset_type':'SID','year':2009,'filename':'CA_SID_2009_CORE.asc.gz', },
    {'state':'CA','file_type':'CORE','dataset_type':'SID','year':2010,'filename':'CA_SID_2010_CORE.asc.gz', },
    {'state':'CA','file_type':'CORE','dataset_type':'SID','year':2011,'filename':'CA_SID_2011_CORE.asc.gz', },
    {'state':'FL','file_type':'CORE','dataset_type':'SID','year':2006,'filename':'FL_SID_2006_CORE.asc.gz','link':'FL_2006_daystoevent.csv.gz'},
    {'state':'FL','file_type':'CORE','dataset_type':'SID','year':2007,'filename':'FL_SID_2007_CORE.asc.gz','link':'FL_2007_daystoevent.csv.gz'},
    {'state':'FL','file_type':'CORE','dataset_type':'SID','year':2008,'filename':'FL_SID_2008_CORE.asc.gz','link':'FL_2008_daystoevent.csv.gz'},
    {'state':'FL','file_type':'CORE','dataset_type':'SID','year':2009,'filename':'FL_SID_2009_CORE.asc.gz'},
    {'state':'FL','file_type':'CORE','dataset_type':'SID','year':2010,'filename':'FL_SID_2010_CORE.asc.gz'},
    {'state':'FL','file_type':'CORE','dataset_type':'SID','year':2011,'filename':'FL_SID_2011_CORE.asc.gz'},
    {'state':'NY','file_type':'CORE','dataset_type':'SID','year':2006,'filename':'NY_SID_2006_CORE.asc.gz','link':'NY_2006_daystoevent.csv.gz'},
    {'state':'NY','file_type':'CORE','dataset_type':'SID','year':2007,'filename':'NY_SID_2007_CORE.asc.gz','link':'NY_2007_daystoevent.csv.gz'},
    {'state':'NY','file_type':'CORE','dataset_type':'SID','year':2008,'filename':'NY_SID_2008_CORE.asc.gz','link':'NY_2008_daystoevent.csv.gz'},
    {'state':'NY','file_type':'CORE','dataset_type':'SID','year':2009,'filename':'NY_SID_2009_CORE.asc.gz'},
    {'state':'NY','file_type':'CORE','dataset_type':'SID','year':2010,'filename':'NY_SID_2010_CORE.asc.gz'},
    {'state':'NY','file_type':'CORE','dataset_type':'SID','year':2011,'filename':'NY_SID_2011_CORE.asc.gz'}
]

class Combiner(object):
    def __init__(self,state,asc_dir,temp_dir,splits=50):
        self.state = state
        self.asc_dir = asc_dir
        self.temp_dir = temp_dir
        self.files = {}
        self.counter = defaultdict(int)
        self.formatter = formats.SASFormat()
        self.files['unlinked'] = gzip.open(self.temp_dir+'unlinked'+'.'+self.state+'.gz','w') # confirm  changes
        self.splits = splits
        for k in range(self.splits):
            self.files[k] = gzip.open(self.temp_dir+str(k)+'.'+self.state+'.gz','w')

    def add_file(self,raw_file):
        if self.state != raw_file['state']:
            raise ValueError
        for visit_link,line in self.generate_lines(raw_file):
            if visit_link < 0:
                self.files['unlinked'].write(line)
            else:
                self.files[hash(visit_link)%self.splits].write(line)

    def combine(self):
        for fh in self.files.itervalues():
            fh.close()
        print "Combining"
        total = 0
        vtotal = 0
        for k in range(self.splits):
            links = defaultdict(str)
            count = 0
            fo = gzip.open(self.temp_dir+str(k)+'.'+self.state+'.gz')
            for line in fo:
                vtotal += 1
                vlink = line.split('_')[0]
                line = line.replace('\n','').replace('\r','')
                if links[vlink]:
                    links[vlink] += '\t'+line
                else:
                    links[vlink] += vlink+'\t'+line
            fo.close()
            fout = gzip.open(self.temp_dir+str(k)+'.'+self.state+'.gz','w')  # confirm  changes
            for line in links.itervalues():
                count += 1
                total += 1
                fout.write(line+'\n')
            fout.close()
            print k,count,total,vtotal

    def generate_lines(self,fdict):
        print fdict
        days_dict = {}
        links_dict = {}
        if 'link' in fdict and fdict['link']:
            print (fdict['link']+" Loading links ")
            fin = gzip.open(self.asc_dir+fdict['link'])
            lines = fin.readlines()
            fin.close()
            print "file read"
            for i,line in enumerate(lines):
                if i:
                    line = line.strip('\r\n').split(',')
                    k = int(line[0].strip())
                    days_dict[k] = int(line[2].strip())
                    links_dict[k] = int(line[1].strip())
            print (fdict['link']+" Links loaded "+str((len(days_dict),len(links_dict))))
        p = self.formatter.get_format_from_filename(fdict['filename'])
        found = 0
        missing = 0
        fin = gzip.open(self.asc_dir+fdict['filename'])
        for i,line in enumerate(fin):
            if links_dict:
                if int(p.get_element('KEY',line)) in links_dict:
                    days_to_event = days_dict[int(p.get_element('KEY',line))]
                    visit_link = links_dict[int(p.get_element('KEY',line))]
                    found += 1
                else:
                    days_to_event = -1
                    visit_link = -1
                    missing += 1
            else:
                days_to_event = int(p.get_element('DaysToEvent',line))
                visit_link =  int(p.get_element('VisitLink',line))
                if visit_link < 0:
                    missing +=1
                else:
                    found += 1
            yield visit_link,'_'.join([str(visit_link),fdict['state'],str(fdict['year']),fdict['dataset_type'],fdict['file_type'],str(days_to_event),line])
        fin.close()
        print found,missing

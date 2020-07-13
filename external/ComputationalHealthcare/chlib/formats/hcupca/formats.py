__author__ = 'aub3'
import re,os,glob
from collections import defaultdict

LMAP = {
('source_dict',5):('sourceh','S_ROUTINE',61,'Routine, birth, etc'),
('source_dict',4):('sourceh','S_COURT',68,'Court/Law enforcement'),
('source_dict',2):('sourceh','S_HOSPITAL',62,'Another hospital'),
('source_dict',3):('sourceh','S_OTHER',65,'Another health facility including LTC'),
('source_dict',1):('sourceh','S_ED',66,'Emergency department'),
('source_dict',-21):('sourceh','S_UNKNOWN',67,'Unknown'),
('source_dict',-8):('sourceh','S_UNKNOWN',67,'Unknown'),
('source_dict',-9):('sourceh','S_UNKNOWN',67,'Unknown'),
('source_dict',-1):('sourceh','S_UNKNOWN',67,'Unknown'),
('disposition_dict',1):('disph','D_ROUTINE',71,'Routine'),
('disposition_dict',2):('disph','D_HOSPITAL',72,'Transfer to short-term hospital'),
('disposition_dict',5):('disph','D_OTHER',76,'Transfer other SNF, ICF, etc.'),
('disposition_dict',6):('disph','D_HOME',74,'Home Health Care'),
('disposition_dict',20):('disph','D_DEATH',75,'Died in hospital'),
('disposition_dict',21):('disph','D_COURT',70,'to court/law enforcement'),
('disposition_dict',99):('disph','D_UNKNOWNALIVE',79,'Unknown, Alive'),
('disposition_dict',-8):('disph','D_UNKNOWN',77,'Unknown'),
('disposition_dict',-9):('disph','D_UNKNOWN',77,'Unknown'),
('disposition_dict',7):('disph','D_AMA',78,'Against medical advice'),
('died_dict',0) : ('deathh','ALIVE',30,'Alive'),
('died_dict',1) : ('deathh','DEAD',31,'Died in hospital'),
('died_dict',-8) : ('deathh','DEATH_UNKNOWN',32,'Unknown'),
('died_dict',-9) : ('deathh','DEATH_UNKNOWN',32,'Unknown'),
('sex_dict',0) : ('sexh','MALE',10,'Male'),
('sex_dict',1) : ('sexh','FEMALE',11,'Female'),
('sex_dict',-9) : ('sexh','SEX_UNKNOWN',12,'Sex unknown'),
('sex_dict',-8) : ('sexh','SEX_UNKNOWN',12,'Sex unknown'),
('sex_dict',-6) : ('sexh','SEX_UNKNOWN',12,'Sex unknown'),
('payer_dict',1):('payerh','MEDICARE',41,'Medicare'),
('payer_dict',2):('payerh','MEDICAID',42,'Medicaid'),
('payer_dict',3):('payerh','PRIVATE',43,'Private insurance'),
('payer_dict',4):('payerh','SELF',44,'Self-pay'),
('payer_dict',5):('payerh','FREE',47,'No charge'),
('payer_dict',6):('payerh','OTHER',45,'Other payer'),
('payer_dict',-8):('payerh','P_UNKNOWN',46,'Unknown payer'),
('payer_dict',-9):('payerh','P_UNKNOWN',46,'Unknown payer'),
('race_dict',1):('raceh','WHITE',51,'White'),
('race_dict',2):('raceh','Black',52,'Black'),
('race_dict',3):('raceh','HISPANIC',53,'Hispanic'),
('race_dict',4):('raceh','ASIAN',54,'Asian or Pacific Islander'),
('race_dict',5):('raceh','NATIVE',55,'Native American'),
('race_dict',6):('raceh','R_OTHER',56,'Other'),
('race_dict',-8):('raceh','R_UNKNOWN',57,'Race missing, unknown'),
('race_dict',-9):('raceh','R_UNKNOWN',57,'Race missing, unknown'),
('dnr_dict',-1):('dnrh','DNR_UNAVAILABLE',83,'DNR unavailable'),
('dnr_dict',0):('dnrh','DNR_NO',80,'No DNR'),
('dnr_dict',1):('dnrh','DNR_YES',81,'DNR'),
('dnr_dict',-8):('dnrh','DNR_UNKNOWN',82,'DNR missing'),
('dnr_dict',-9):('dnrh','DNR_UNKNOWN',82,'DNR missing'),
('pzip_dict',1):('pziph','Z_FIRST',101,'First income quartile'),
('pzip_dict',2):('pziph','Z_SECOND',102,'First income quartile'),
('pzip_dict',3):('pziph','Z_THIRD',103,'First income quartile'),
('pzip_dict',4):('pziph','Z_FOURTH',104,'First income quartile'),
('pzip_dict',-9):('pziph','Z_UNKNOWN',105,'Unknown income quartile'),
('pzip_dict',-8):('pziph','Z_UNKNOWN',105,'Unknown income quartile'),
}

class SASFormat(object):
    def __init__(self,path=os.path.join(os.path.dirname(__file__),'formats')):
        self.columns = defaultdict(dict)
        self.length = {}
        for fname in glob.glob(path+'/*.sas'):
            filename = os.path.split(fname)[-1]
            state,dataset_type,year,file_type = filename.split('.')[0].split('_')
            year = int(year)
            columns = {}
            post_input = False
            lines = file(fname).readlines()
            for i,line in enumerate(lines):
                if 'LRECL = ' in line:
                    length = int(line.strip().split('LRECL = ')[1].strip(';').split(' ')[0])
                if line.strip() == 'INPUT':
                    post_input = True
                elif post_input and line.strip().startswith('@'):
                    start,element,element_type = filter(None,line.strip().replace('@','').split(' '))
                    start = int(start)
                    if element in columns:
                        raise ValueError,"Element repeated "+element
                    else:
                        columns[element] = [start-1,None,element_type] # to fix 1 index
            sorted_columns = sorted([(v[0],k) for k,v in columns.iteritems()])
            for i, k_v in enumerate(sorted_columns):
                k, v = k_v
                if i+1 != len(sorted_columns):
                    columns[v][1] = sorted_columns[i+1][0]
                else:
                    columns[v][1] = None
            for k,v in columns.iteritems():
                self.columns[(state,dataset_type,year,file_type)][k] = v
            self.length[(state,dataset_type,year,file_type)] = length

    def store(self):
        fh = open('fields.csv','w')
        for ftuple,element_dict in self.columns.iteritems():
            for k,v in element_dict.iteritems():
                fh.write('\t'.join([str(s) for s in ftuple]+[str(k),]+[str(s) for s in v])+'\n')
        fh.close()

    def element_types(self):
        temp = defaultdict(set)
        for k,v in self.columns.iteritems():
            for e,t in v.iteritems():
                temp[e].add(t[2])
        return temp

    def element_lengths(self):
        temp = defaultdict(set)
        for k,v in self.columns.iteritems():
            for e,t in v.iteritems():
                temp[e].add(t[1]-t[0])
        return temp

    def get_format_from_filename(self,fname):
        state,dataset_type,year,file_type = fname.split('/')[-1].split('.')[0].split('_')
        year = int(year)
        return self.get_format(state,dataset_type,year,file_type)

    def get_format(self,state,dataset_type,year,file_type):
        ftuple = (state,dataset_type,year,file_type)
        if ftuple in self.columns:
            return Parser(self.columns[ftuple])
        else:
            raise ValueError,str(ftuple) + "count not be found"


class Parser(object):
    def __init__(self,columns):
        self.columns = columns

    def get_element(self,element,line):
        if element in self.columns:
            return line[self.columns[element][0]:self.columns[element][1]].strip()


if __name__ == '__main__':
    test = SASFormat()
    test.store()
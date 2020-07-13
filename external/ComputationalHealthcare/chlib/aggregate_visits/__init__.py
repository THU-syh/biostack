__author__ = 'aub3'
from ..entity import penums_pb2 as enums
from ..entity import aggregate
from .. import codes
from ..entity import aggregate as ag
from ..entity.pn1_pb2 import PEntry,DX,DRG,NONE
from ..entity.presentation import SUBSET_PLOT_TABLES
from ..entity import presentation
from collections import defaultdict
import math,json,logging
from ..entity import penums_pb2 as enums
from ..entity import stream_pb
from ..entity.pvisit_pb2 import Visit
from ..entity.aggregate import process_exclusions
from collections import defaultdict

NAME = 'Procedure Analytics'

SPLIT_HASH_MAX = 8 # how to parallelize split process
BASE_URL = "/N1"
MIN_PDAY = -4
DELTA_MIN = 0
DELTA_MAX = 350
AGE_MIN = 20
N1_POLICY = aggregate.Policy(base=10,min_count=20,min_subset=100,min_hospital=2)


SOURCE_EVENTS = {
    'N1_a':'Admission all sources',
    'N1_e':'Admission via ED',
    'N1_s':'Admitted via Long Term Care, Skilled Nursing Facility etc',
    'N1_h':'Transferred in from Hospital',
    'N1_r':'Routine admission',
}


SOURCE_MAP = {
    enums.S_ED:'N1_e',
    enums.S_HOSPITAL:'N1_h',
    enums.S_ROUTINE:'N1_r',
    enums.S_SNF:'N1_s',
}

DISP_MAP = {
    enums.D_ROUTINE:'N1_f',
    enums.D_HOSPITAL:'N1_n',
    enums.D_SNF:'N1_l',
    enums.D_HOME:'N1_c',
    enums.D_DEATH:'N1_m',
}

DISP_EVENTS = {
    'N1_x':'Discharge, Transferred or Died',
    'N1_d':'Discharged or Transferred alive',
    'N1_f':'Discharged to home',
    'N1_n':'Transferred to Hospital',
    'N1_l':'Transferred to SNF or LTC',
    'N1_c':'Discharged to Home Health Care',
    'N1_m':'Died in hospital',
}


class N1Coder(codes.Coder):
    def __init__(self):
        self.STRINGS = {}
        for k,v in SOURCE_EVENTS.iteritems():
            self.STRINGS[k] = v
        for k,v in DISP_EVENTS.iteritems():
            self.STRINGS[k] = v
        super(N1Coder, self).__init__()

    def __getitem__(self, item):
        if item in self.STRINGS:
            return self.STRINGS[item]
        else:
            return super(N1Coder,self).__getitem__(item)


def default_entry():
    return Entry()


def age_cat(age):
    if 20 >= age >= 0:
        return "Age < 20 year "
    elif 45 >= age > 20:
        return "45 >= Age > 20 "
    elif 65 >= age > 45:
        return "65 >= Age > 45 "
    elif 85 >= age > 65:
        return "85 >= Age > 65 "
    elif age > 85:
        return "Age > 85 "


class Entry(object):
    """
    required string key = 1;
    optional string dataset = 2;
    required ENTRYTYPE etype = 3;
    required string dx = 4;
    required string initial = 5;
    required string sub = 6;
    required IntHist deltah = 7;
    required AGG stats = 8;
    repeated ISubset delta_subset = 9;
    repeated ISubset age_subset = 10;
    repeated ISubset year_subset = 11;
    repeated RSEntry state_subset = 12;
    optional string dataset_s = 16;
    """
    def __init__(self,state_subset_flag = True):
        self.obj = PEntry()
        self.agg = None
        self.delta_counter = defaultdict(int)
        self.delta_subset = {}
        self.state_subset = {}
        self.year_subset = {}
        self.age_subset = {}
        self.state_subset_flag = state_subset_flag

    def ParseFromString(self,s):
        self.obj.ParseFromString(s)

    def init_compute(self,dx,initial,sub,dataset,policy,pediatric = False):
        self.agg = ag.Aggregate()
        self.obj.dx = dx
        self.obj.initial = initial
        self.obj.sub = sub
        self.obj.dataset = dataset
        self.dx = dx
        self.initial = initial
        self.sub = sub
        self.dataset = dataset
        self.base = policy.base
        self.min_count = policy.min_count
        self.min_hospital = policy.min_hospital
        self.min_subset = policy.min_subset
        self.policy = policy
        self.pediatric = pediatric
        if dx and (dx.startswith('DG') or dx.startswith('G')):
            self.obj.etype = DRG
        elif dx and dx.startswith('D'):
            self.obj.etype = DX
        else:
            self.obj.etype = NONE
        self.key = '_'.join([dx,initial,sub])
        self.obj.key = self.key
        self.agg.init_compute(self.key,dataset,self.policy)


    def add(self,v,delta):
        self.delta_counter[delta] += 1
        if delta in self.delta_subset:
            self.delta_subset[delta].add(v)
        else:
            self.delta_subset[delta] = ag.Aggregate(mini=True)
            self.delta_subset[delta].init_compute(self.key,"Delta "+str(delta)+"  "+self.dataset,self.policy)
        if not v.year in self.year_subset:
            self.year_subset[v.year] = ag.Aggregate(mini=True)
            self.year_subset[v.year].init_compute(self.key,"Year "+str(v.year)+"  "+self.dataset,self.policy)
        if not age_cat(v.age) in self.age_subset:
            self.age_subset[age_cat(v.age)] = ag.Aggregate(mini=True)
            self.age_subset[age_cat(v.age)].init_compute(self.key,"Age "+age_cat(v.age)+"  "+self.dataset,self.policy)
        if self.state_subset_flag:
            if not v.state in self.state_subset:
                self.state_subset[v.state] = Entry(state_subset_flag=False)
                self.state_subset[v.state].init_compute(self.dx,self.initial,self.sub,"State "+v.state+"  "+self.dataset,self.policy,pediatric=self.pediatric)
            self.state_subset[v.state].add(v,delta)
        self.year_subset[v.year].add(v)
        self.age_subset[age_cat(v.age)].add(v)
        self.agg.add(v)


    def plot_data(self):
        # delta_subset_joint ={}
        # for t in self.obj.deltah.h:
        #     delta_subset_joint[t.k] = {'delta':t.k,'visits':t.v,'denominator':t.v}
        return [(t.k,t.v) for t in self.obj.deltah.h]

    def age_plot(self):
        age_plot_data = { k:0 for k in range(20,100)}
        age_plot_data.update({t.k:t.v for t in self.obj.stats.ageh.h})
        return age_plot_data.items()

    def delta_subset_plot(self):
        delta_count = {t.k:t.v for t in self.obj.deltah.h }
        delta_plot_data = {}
        for temp in self.obj.delta_subset:
            delta = temp.k
            if delta in delta_count and delta_count[delta]:
                for attr in SUBSET_PLOT_TABLES:
                    for e in getattr(temp.subset,attr):
                        if attr not in delta_plot_data:
                            delta_plot_data[attr] = {}
                        if e.k not in delta_plot_data[attr]:
                            delta_plot_data[attr][e.k] = []
                        delta_plot_data[attr][e.k].append([delta,round(100.0*e.v/delta_count[delta],2)])
        return delta_plot_data

    def los_plot(self):
        los_plot_data = { k:0 for k in range(0,max([t.k for t in self.obj.stats.losh.h]+[0,])+5)}
        los_plot_data.update({t.k:t.v for t in self.obj.stats.losh.h})
        return los_plot_data.items()


    def end_compute(self):
        if self.agg.end_compute():
            self.obj.pediatric = self.pediatric
            self.obj.stats.CopyFrom(self.agg.obj)
            if self.state_subset_flag:
                for state,state_entry in self.state_subset.iteritems():
                    if state_entry.end_compute():
                        temp = self.obj.state_subset.add()
                        temp.k = state
                        temp.s = state
                        temp.subset.CopyFrom(state_entry.obj)
            for delta,delta_agg in self.delta_subset.iteritems():
                if delta_agg.end_compute():
                    temp = self.obj.delta_subset.add()
                    temp.k = delta
                    temp.subset.CopyFrom(delta_agg.obj)
            for year,year_agg in self.year_subset.iteritems():
                if year_agg.end_compute():
                    temp = self.obj.year_subset.add()
                    temp.k = year
                    temp.subset.CopyFrom(year_agg.obj)
            for age,age_agg in self.age_subset.iteritems():
                if age_agg.end_compute():
                    temp = self.obj.age_subset.add()
                    temp.k = age
                    temp.subset.CopyFrom(age_agg.obj)
            mean,median,fq,tq = ag.compute_stats(self.delta_counter)
            self.obj.deltah.median = int(round(median))
            self.obj.deltah.mean = round(mean,2)
            self.obj.deltah.fq = int(round(fq))
            self.obj.deltah.tq = int(round(tq))
            for value,c in self.delta_counter.iteritems():
                if value >= 0:
                    temp = self.obj.deltah.h.add()
                    temp.k = value
                    temp.v = int(self.base*int(math.floor(c/float(self.base)))) if c > self.min_count else 0
            return True

    def SerializeToString(self):
        return self.obj.SerializeToString()

    def __str__(self):
        return self.obj.__str__()


def aggregate_events(code,visits,dataset,result_dir,reduce_mode_mini=True):
    code_type_pr = code.startswith('P')
    negative_delta_count = defaultdict(int)
    counter = defaultdict(int)
    exclusion_counter = defaultdict(int)
    key_list = defaultdict(list)
    aggregator = {}
    pediatric_aggregator = {}
    logging.info("Start "+code+" reduce_mode_mini "+str(reduce_mode_mini))
    i = 0
    for visit_obj in visits:
        i += 1
        for dx,initial,sub,delta in get_ordered_pairs(visit_obj,negative_delta_count):
            if dx == code:
                counter[(dx,initial,sub)] += 1
                key_list[i].append(((dx,initial,sub),delta))
            elif reduce_mode_mini:
                if dx == '' and (code == initial or (initial.startswith('N1') and code == sub)):
                    counter[(dx,initial,sub)] += 1
                    key_list[i].append(((dx,initial,sub),delta))
            else:
                if code == initial or code == sub:
                    counter[(dx,initial,sub)] += 1
                    key_list[i].append(((dx,initial,sub),delta))
    logging.info("Finished counting counter length "+str(len(counter)))
    i = 0
    for visit_obj in visits:
        i += 1
        adult_pediatric_flag = check_criteria(visit_obj,exclusion_counter)
        if adult_pediatric_flag == 0: # Adult
            for key_tuple,delta in key_list[i]:
                if (key_tuple[0] == code or code_type_pr) and counter[key_tuple] > N1_POLICY.min_subset:
                    if not(key_tuple in aggregator):
                        aggregator[key_tuple] = Entry()
                        dx,initial,sub = key_tuple
                        aggregator[key_tuple].init_compute(dx,initial,sub,dataset,N1_POLICY)
                    aggregator[key_tuple].add(visit_obj,delta)
        elif adult_pediatric_flag == 1: # Pediatric
            for key_tuple,delta in key_list[i]:
                if (key_tuple[0] == code or code_type_pr) and counter[key_tuple] > N1_POLICY.min_subset:
                    if not(key_tuple in pediatric_aggregator):
                        pediatric_aggregator[key_tuple] = Entry()
                        dx,initial,sub = key_tuple
                        pediatric_aggregator[key_tuple].init_compute(dx,initial,sub,dataset,N1_POLICY,pediatric=True)
                    pediatric_aggregator[key_tuple].add(visit_obj,delta)
        else:
            continue

    logging.info("Finished aggregation aggregate length "+str(len(aggregator)))
    updates = []
    for k,v in aggregator.iteritems():
        if v.end_compute():
            fname = result_dir+"/A"+v.key+".n1protoentry"
            foutput = open(fname,'w')
            foutput.write(v.SerializeToString())
            foutput.close()
            updates.append({
                'dx':v.dx,
                "initial":v.initial,
                'sub':v.sub,
                'filename':fname.split('/')[-1],
                'count':v.obj.stats.count,
                'split_code':code,
                'delta_median':v.obj.deltah.median,
                'key':'_'.join(['A',v.dx,v.initial,v.sub]),
                'pediatric':False
                })
    for k,v in pediatric_aggregator.iteritems():
        if v.end_compute():
            fname = result_dir+"/P"+v.key+".n1protoentry"
            foutput = open(fname,'w')
            foutput.write(v.SerializeToString())
            foutput.close()
            updates.append({
                'dx':v.dx,
                "initial":v.initial,
                'sub':v.sub,
                'filename':fname.split('/')[-1],
                'count':v.obj.stats.count,
                'split_code':code,
                'delta_median':v.obj.deltah.median,
                'key':'_'.join(['P',v.dx,v.initial,v.sub]),
                'pediatric':True
                })
    excluded = process_exclusions(exclusion_counter,N1_POLICY)
    result = {'updates':updates,'excluded':excluded,'code':code}
    with open("{}/{}.inpatient_events_meta.json".format(result_dir,code),'w') as fh:
        json.dump(result,fh)
    return result


def add_events(v):
    source = ['N1_a',]
    disposition = ['N1_x',]
    if v.source in SOURCE_MAP:
        source.append(SOURCE_MAP[v.source])
    if v.disposition in DISP_MAP:
        disposition.append(DISP_MAP[v.disposition])
    if v.disposition != enums.D_DEATH:
        disposition.append('N1_d')
    return source,disposition


def check_criteria(v,ex_counter):
    ex_counter['total'] += 1
    if v.los < 0:
        ex_counter['excluded'] += 1
        ex_counter[(1,'LOS < 0')] += 1
        return -1 # exclude
    elif v.age < 0:
        ex_counter['excluded'] += 1
        ex_counter[(2,'Age < 0')] += 1
        return -1 # exclude
    elif v.age <= AGE_MIN:
        ex_counter['selected_pediatric'] += 1
        return 1 # pediatric
    ex_counter['selected'] += 1
    return 0 # adult


def get_ordered_pairs(v,negative_delta_count):
    # for dx,initial,sub,delta in get_ordered_pairs(visit_obj):
    pairs,edges = [], []
    source,disposition = add_events(v)
    pr = [(k.pday,k.pcode) for k in v.prs if k.pday >= MIN_PDAY]
    pr = sorted(pr)
    if pr:
        for i,initial in enumerate(pr):
            for sub in pr[i+1:]:
                edges.append((initial,sub))
    for k,l in edges:
        if k == l: # needed to stop double counting in case of (0,p),(0,p)
            delta = 0
            pairs.append((k[1],l[1],delta))
        elif k[0] == l[0]:
            delta = 0
            pairs.append((k[1],l[1],delta))
            pairs.append((l[1],k[1],delta))
        else:
            delta = (l[0]-k[0])
            if delta < DELTA_MIN:
                negative_delta_count[(k[1],l[1])] += 1
            elif delta < DELTA_MAX:
                pairs.append((k[1],l[1],delta))
    for s in source:
        for sub in pr:
            pairs.append((s,sub[1],sub[0]))
    for d in disposition:
        for initial in pr:
            pairs.append((initial[1],d,v.los-initial[0])) # catch negatives and count them
    for s in source:
        for d in disposition:
            pairs.append((s,d,v.los))
    for dx in ["",v.drg,v.primary_diagnosis]:
        for initial,sub,delta in pairs:
            yield dx,initial,sub,delta

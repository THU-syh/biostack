import json,math,datetime,os,logging
from collections import defaultdict
from ..entity import pn3_pb2
from ..entity import aggregate as ag
from ..entity import enums,aggregate
from .. import codes
from ..entity import enums
from ..entity.visit import VisitEdge,pvisit_pb2
from ..entity.aggregate import sanitize,process_exclusions
from collections import defaultdict

NAME = 'Revisit Analytics'
SPLIT_HASH_MAX = 8
AGE_MIN = 19
N3_POLICY = aggregate.Policy(min_count=20,base=10,min_hospital=2,min_subset=200)


EVENTS = {
    (0,enums.IP):('N3_IP_0','Transfer hospitalizations and same day readmissions'),
    (7,enums.IP):('N3_IP_7','Readmission in 7 days '),
    (30,enums.IP):('N3_IP_30','Readmission in 30 days '),
    (60,enums.IP):('N3_IP_60','Readmission in 60 days '),
    (90,enums.IP):('N3_IP_90','Readmission in 90 days '),
    (365,enums.IP):('N3_IP_365','Readmission in 365 days '),
    (0,enums.AS):('N3_AS_0','Transfer to Ambulatory surgery and same day revisit'),
    (7,enums.AS):('N3_AS_7','Ambulatory Surgery in 7 days '),
    (30,enums.AS):('N3_AS_30','Ambulatory Surgery in 30 days '),
    (60,enums.AS):('N3_AS_60','Ambulatory Surgery in 60 days '),
    (90,enums.AS):('N3_AS_90','Ambulatory Surgery in 90 days'),
    (365,enums.AS):('N3_AS_365','Ambulatory Surgery in 365 days '),
    (0,enums.ED):('N3_ED_0','Transfer to ED and same day revisit'),
    (7,enums.ED):('N3_ED_7','ED  visits in 7 days '),
    (30,enums.ED):('N3_ED_30','ED  visits in 30 days '),
    (60,enums.ED):('N3_ED_60','ED  visits in 60 days '),
    (90,enums.ED):('N3_ED_90','ED  visits in 90 days '),
    (365,enums.ED):('N3_ED_365','ED  visits in 365 days ')
}


class N3Coder(codes.Coder):
    def __init__(self):
        self.STRINGS = {}
        for v in EVENTS.itervalues():
            self.STRINGS[v[0]] = v[1]
        super(N3Coder, self).__init__()

    def __getitem__(self, item):
        if not isinstance(item,int):
            if item.startswith('N3CPT_AS'):
                return "Procedure during ambulatory surgery visits :"+super(N3Coder,self).__getitem__(item.split('N3CPT_AS_')[1])
            elif item.startswith('N3CPT_ED'):
                return "Procedure during ED visits :"+ super(N3Coder,self).__getitem__(item.split('N3CPT_ED_')[1])
            elif item.startswith('N3C_AS'):
                return "Procedure during ambulatory surgery visits :"+super(N3Coder,self).__getitem__(item.split('N3C_AS_')[1])
            elif item.startswith('N3C_ED'):
                return "Procedure during ED visits :"+ super(N3Coder,self).__getitem__(item.split('N3C_ED_')[1])
            elif item.startswith('N3DX_IP'):
                return "Primary Diagnosis during inpatient visits :" +  super(N3Coder,self).__getitem__(item.split('N3DX_IP_')[1])
            elif item.startswith('N3DX_ED'):
                return "Primary Diagnosis during ED visits :" +  super(N3Coder,self).__getitem__(item.split('N3DX_ED_')[1])
            elif item.startswith('N3DX_AS'):
                return "Primary Diagnosis during ambulatory surgery visits :" + super(N3Coder,self).__getitem__(item.split('N3DX_AS_')[1])
        if item in self.STRINGS:
            return self.STRINGS[item]
        else:
            return super(N3Coder,self).__getitem__(item)


class Edge(object):
    """
    required string initial_key = 1;
    required string sub_key = 2;
    optional string dataset = 3;
    optional int32 delta = 8;
    required AGG initial = 4;
    required AGG sub = 5;
    required IntHist deltah = 6;
    repeated Exclusion provenance = 7;
    required KEYTYPE initial_ktype = 9;
    required KEYTYPE sub_ktype = 10;
    """
    def __init__(self):
        self.obj = pn3_pb2.VEntry()

    def ParseFromString(self,s):
        self.obj.ParseFromString(s)

    def init_compute(self,initial_key,sub_key,dataset,policy):
        self.initial = ag.Aggregate()
        self.sub = ag.Aggregate()
        self.obj.initial_key = initial_key
        self.obj.sub_key = sub_key
        self.obj.dataset = dataset
        self.policy = policy
        self.dataset = dataset
        self.base = policy.base
        self.min_count = policy.min_count
        self.min_hospital = policy.min_hospital
        self.delta_counter = defaultdict(int)
        self.delta_counter_week = defaultdict(int)
        if initial_key.startswith('DG'):
            self.obj.initial_ktype = pn3_pb2.N3_DRG
        elif initial_key.startswith('N3DX'):
            self.obj.initial_ktype = pn3_pb2.N3_DX
        elif initial_key.startswith('P'):
            self.obj.initial_ktype = pn3_pb2.N3_ICDPR
        elif initial_key.startswith('N3CPT'):
            self.obj.initial_ktype = pn3_pb2.N3_CPT
        elif initial_key.startswith('N3'):
            self.obj.initial_ktype = pn3_pb2.N3_ALL
        if sub_key.startswith('DG'):
            self.obj.sub_ktype = pn3_pb2.N3_DRG
        elif sub_key.startswith('D'):
            self.obj.sub_ktype = pn3_pb2.N3_DX
        elif sub_key.startswith('P'):
            self.obj.sub_ktype = pn3_pb2.N3_ICDPR
        elif sub_key.startswith('C'):
            self.obj.sub_ktype = pn3_pb2.N3_CPT
        elif sub_key.startswith('N3'):
            self.obj.sub_ktype = pn3_pb2.N3_ALL
        self.key = '_'.join([initial_key,sub_key])
        self.obj.key = self.key
        self.initial.init_compute(initial_key,dataset,policy)
        self.sub.init_compute(sub_key,dataset,policy)

    def add(self,initial,sub,delta):
        self.delta_counter[delta] += 1
        self.delta_counter_week[int(math.floor(delta/7.0))] += 1
        self.initial.add(initial)
        self.sub.add(sub)

    def end_compute(self):
        if self.initial.end_compute() and self.sub.end_compute():
            self.obj.initial.CopyFrom(self.initial.obj)
            self.obj.sub.CopyFrom(self.sub.obj)
            mean,median,fq,tq = ag.compute_stats(self.delta_counter)
            self.obj.deltah.median = int(round(median))
            self.obj.deltah.mean = round(mean,2)
            self.obj.deltah.fq = int(round(fq))
            self.obj.deltah.tq = int(round(tq))
            for value,c in self.delta_counter.iteritems():
                if value >= 0:
                    temp = self.obj.deltah.h.add()
                    temp.k = value
                    temp.v = int(self.base*int(math.floor(c/float(self.base)))) if c > self.policy.min_count else 0
            mean,median,fq,tq = ag.compute_stats(self.delta_counter_week)
            self.obj.deltaweekh.median = int(round(median))
            self.obj.deltaweekh.mean = round(mean,2)
            self.obj.deltaweekh.fq = int(round(fq))
            self.obj.deltaweekh.tq = int(round(tq))
            for value,c in self.delta_counter_week.iteritems():
                if value >= 0:
                    temp = self.obj.deltaweekh.h.add()
                    temp.k = value
                    temp.v = int(self.base*int(math.floor(c/float(self.base)))) if c > self.policy.min_count else 0
            return True
        return False

    def SerializeToString(self):
        return self.obj.SerializeToString()

    def __str__(self):
        return self.obj.__str__()

    def plot_data(self):
        # delta_subset_joint ={}
        # for t in self.obj.deltah.h:
        #     delta_subset_joint[t.k] = {'delta':t.k,'visits':t.v,'denominator':t.v}
        return [(t.k,t.v) for t in self.obj.deltah.h]

    def week_plot_data(self):
        return [(t.k,t.v) for t in self.obj.deltaweekh.h]

    def age_plot_initial(self):
        age_plot_data = { k:0 for k in range(20,100)}
        age_plot_data.update({t.k:t.v for t in self.obj.initial.ageh.h})
        return age_plot_data.items()

    def age_plot_sub(self):
        age_plot_data = { k:0 for k in range(20,100)}
        age_plot_data.update({t.k:t.v for t in self.obj.sub.ageh.h})
        return age_plot_data.items()

    def los_plot_initial(self):
        los_plot_data = { k:0 for k in range(0,max([t.k for t in self.obj.initial.losh.h]+[0,])+5)}
        los_plot_data.update({t.k:t.v for t in self.obj.initial.losh.h})
        return los_plot_data.items()

    def los_plot_sub(self):
        los_plot_data = { k:0 for k in range(0,max([t.k for t in self.obj.sub.losh.h]+[0,])+5)}
        los_plot_data.update({t.k:t.v for t in self.obj.sub.losh.h})
        return los_plot_data.items()


class Node(object):
    """
    required string key = 1;
    required KEYTYPE ktype = 2;
    optional string dataset = 3;
    required AGG all = 4;
    required AGG discharged = 5;
    required AGG transferred = 6;
    required AGG died = 7;
    repeated Exclusion provenance = 8;
    """
    def __init__(self):
        self.obj = pn3_pb2.VNode()

    def ParseFromString(self,s):
        self.obj.ParseFromString(s)

    def init_compute(self,key,dataset,policy,linked):
        self.all = ag.Aggregate()
        self.all.init_compute(key,dataset,policy)
        self.discharged = ag.Aggregate()
        self.discharged.init_compute("{},discharged".format(key),dataset,policy)
        self.transferred = ag.Aggregate()
        self.transferred.init_compute("{},transferred".format(key),dataset,policy)
        self.died = ag.Aggregate()
        self.died.init_compute("{},died".format(key),dataset,policy)
        self.obj.key = key
        self.obj.dataset = dataset
        self.linked = linked
        self.dataset = dataset
        self.base = policy.base
        self.min_count = policy.min_count
        self.min_hospital = policy.min_hospital
        if key.startswith('DG'):
             self.obj.ktype = pn3_pb2.N3_DRG
        elif key.startswith('N3D'):
             self.obj.ktype = pn3_pb2.N3_DX
        elif key.startswith('N3C'):
             self.obj.ktype = pn3_pb2.N3_CPT
        elif key.startswith('P'):
             self.obj.ktype = pn3_pb2.N3_ICDPR
        else:
            raise NotImplementedError,key
        self.key = key
        self.obj.key = self.key
        self.obj.linked = linked
    def add(self,index):
        self.all.add(index)
        if index.death == enums.DEAD:
            self.died.add(index)
        elif index.disposition == enums.D_HOSPITAL:
            self.transferred.add(index)
        else:
            self.discharged.add(index)

    def end_compute(self):
        if self.all.end_compute():
            self.obj.all.CopyFrom(self.all.obj)
            if self.died.end_compute():
                self.obj.died.CopyFrom(self.died.obj)
            if self.transferred.end_compute():
                self.obj.transferred.CopyFrom(self.transferred.obj)
            if self.discharged.end_compute():
                self.obj.discharged.CopyFrom(self.discharged.obj)
            return True
        return False

    def SerializeToString(self):
        return self.obj.SerializeToString()

    def __str__(self):
        return self.obj.__str__()

    def los_plot_discharged(self):
        los_plot_data = { k:0 for k in range(0,max([t.k for t in self.obj.discharged.losh.h]+[0,])+5)}
        los_plot_data.update({t.k:t.v for t in self.obj.discharged.losh.h})
        return los_plot_data.items()

    def age_plot_discharged(self):
        age_plot_data = { k:0 for k in range(20,100)}
        age_plot_data.update({t.k:t.v for t in self.obj.discharged.ageh.h})
        return age_plot_data.items()


def aggregate_revisits(code, unlinked_visits, index_visits, revisit_edges, dataset_id, result_dir, last_year_exclude, reduce_mode_mini=True):
    aggregator = {}
    try:
        os.makedirs(result_dir)
    except OSError:
        pass
    logging.info("Start {}, reduce mode {} ".format(code,reduce_mode_mini))
    unlinked_updates, ex_unlinked = process_unlinked(code, dataset_id, unlinked_visits, result_dir, last_year_exclude)
    index_updates, ex_index = process_index(code,dataset_id,index_visits,result_dir,last_year_exclude)
    if revisit_edges:
        edge_updates,ex_edge = process_edges(code,dataset_id,revisit_edges,result_dir,last_year_exclude)
    else:
        edge_updates = None
        ex_edge = defaultdict(int)
    logging.info("Finished edge_aggregation aggregate length "+str(len(aggregator)))
    exclusions_unlinked = process_exclusions(ex_unlinked,N3_POLICY)
    exclusions_index = process_exclusions(ex_index,N3_POLICY)
    exclusions_edge = process_exclusions(ex_edge,N3_POLICY)
    result = {'unlinked_updates':unlinked_updates,
                'index_updates':index_updates,
                'edge_updates':edge_updates,
                'exclusions_unlinked':exclusions_unlinked,
                'exclusions_index':exclusions_index,
                'exclusions_edge':exclusions_edge,
                'code': code,}
    with open("{}/{}.revisit_meta.json".format(result_dir,code),'w') as fh:
        json.dump(result,fh)
    return result


def process_unlinked(code,dataset,unlinked,fout,last_year_exclude=-1):
    exclusion_counter = defaultdict(int)
    node = Node()
    node.init_compute(code,dataset,N3_POLICY,linked=False)
    for i,v in enumerate(unlinked):
        if check_criteria_node(v,exclusion_counter,-1):
            node.add(v)
    if node.end_compute():
        fname = fout+"/"+node.key+".n3_node"
        logging.info(fname)
        foutput = open(fname,'w')
        foutput.write(node.SerializeToString())
        foutput.close()
        updates = [{"entry_type":"node",
                    "initial":node.obj.key,
                    "index":node.obj.key,
                    'filename':node.key+".n3_node",
                    'count':node.obj.all.count,
                    'split_code':code,
                    'linked':False,
                    'key':node.obj.key},]
    else:
        updates = []
    return updates,exclusion_counter


def process_index(code,dataset,index,fout,last_year_exclude):
    exclusion_counter = defaultdict(int)
    node = Node()
    node.init_compute(code,dataset,N3_POLICY,linked=True)
    for i,v in enumerate(index):
        if check_criteria_node(v,exclusion_counter,last_year_exclude):
            node.add(v)
    if node.end_compute():
        fname = fout+"/"+node.key+".n3_node"
        logging.info(fname)
        foutput = open(fname,'w')
        foutput.write(node.SerializeToString())
        foutput.close()
        updates = [{"entry_type":"node",
                    "initial":node.obj.key,
                    "index":node.obj.key,
                    'filename':node.key+".n3_node",
                    'count':node.obj.all.count,
                    'split_code':code,
                    'linked':True,
                    'key':node.obj.key},]
    else:
        updates = []
    return updates,exclusion_counter


def process_edges(code,dataset,edges,fout,last_year_exclude):
    exclusion_counter = defaultdict(int)
    counter = defaultdict(list)
    edge_aggregator = {}
    for i,e in enumerate(edges):
        if check_criteria(e,exclusion_counter,last_year_exclude):
            for initial_key,sub_key,delta in get_keys(e,code,exclusion_counter):
                counter[i].append((initial_key,sub_key,delta))
    for i,e in enumerate(edges):
        for initial_key,sub_key,delta in counter[i]:
            ktuple = (initial_key,sub_key)
            if not ktuple in edge_aggregator:
                edge_aggregator[ktuple] = Edge()
                edge_aggregator[ktuple].init_compute(initial_key,sub_key,dataset,N3_POLICY)
            edge_aggregator[ktuple].add(e.initial,e.sub,delta)
    updates = []
    for k,v in edge_aggregator.iteritems():
        if v.end_compute():
            fname = fout+"/"+v.key+".n3_edge"
            foutput = open(fname,'w')
            foutput.write(v.SerializeToString())
            foutput.close()
            updates.append({"entry_type":"edge","initial":v.obj.initial_key,'sub':v.obj.sub_key,'filename':v.key+".n3_edge",'count':v.obj.initial.count,'split_code':code,'key':v.obj.key})
    return updates,exclusion_counter


def get_keys(e,code,counter):
    delta = e.sub.day - (e.initial.day + e.initial.los)
    if delta >= 0:
        if e.sub.vtype == enums.IP:
            yield code,'N3DX_IP_'+e.sub.primary_diagnosis,delta
        elif e.sub.vtype == enums.AS:
            yield code,'N3DX_AS_'+e.sub.primary_diagnosis,delta
        elif e.sub.vtype == enums.ED:
            yield code,'N3DX_ED_'+e.sub.primary_diagnosis,delta
        for sub_key in [e.sub.drg,]+[c.pcode for c in e.sub.prs]:
            if sub_key.startswith('C') and e.sub.vtype == enums.ED:
                yield code,'N3CPT_ED_'+sub_key,delta
            elif sub_key.startswith('C') and e.sub.vtype == enums.AS:
                yield code,'N3CPT_AS_'+sub_key,delta
            elif sub_key != 'DG-1':
                yield code,sub_key,delta
        if delta == 0:
            for dtuple,event_key in EVENTS.iteritems():
                dlimit,dvtype = dtuple
                if dlimit == 0 and delta == 0 and e.sub.vtype == dvtype:
                    yield code,event_key[0],delta
        else:
            for dtuple,event_key in EVENTS.iteritems():
                dlimit,dvtype = dtuple
                if dlimit > 0 and delta < dlimit and e.sub.vtype == dvtype:
                    yield code,event_key[0],delta
    else:
        counter['negative_delta_exclude'] += 1


def check_criteria(e,counter,exclude_year):
    delta = e.sub.day - (e.initial.day + e.initial.los)
    counter['total'] += 1
    if e.initial.age <= AGE_MIN:
        counter['excluded'] += 1
        counter[(1,"Age < {}".format(AGE_MIN))] += 1
        return False
    elif e.initial.los < 0:
        counter['excluded'] += 1
        counter[(2,"Initial LOS < 0")] += 1
        return False
    elif e.sub.los < 0:
        counter['excluded'] += 1
        counter[(3,"Sub LOS < 0")] += 1
        return False
    elif e.initial.year == exclude_year:
        counter['excluded'] += 1
        counter[(4,"Initial visit during {}".format(exclude_year))] += 1
        return False
    elif delta < 0:
        counter['excluded'] += 1
        counter[(5,"Delta < 0")] += 1
        return False
    elif delta > 365:
        counter['excluded'] += 1
        counter[(6,"Delta > 365 days")] += 1
        return False
    else:
        counter['selected'] += 1
        return True


def check_criteria_node(v,counter,exclude_year):
    counter['total'] += 1
    if v.age <= AGE_MIN:
        counter['excluded'] += 1
        counter[(1,"Age < {}".format(AGE_MIN))] += 1
        return False
    elif v.los < 0:
        counter['excluded'] += 1
        counter[(2,"Index LOS < 0")] += 1
        return False
    elif v.year == exclude_year:
        counter['excluded'] += 1
        counter[(3,"Index visit during {}".format(exclude_year))] += 1
    else:
        counter['selected'] += 1
        return True

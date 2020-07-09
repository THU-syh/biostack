from .. import codes
from ..entity.visit import Patient
from ..entity import enums,aggregate
from ..entity.pn4_pb2 import NEntry
from ..entity.aggregate import sanitize,process_exclusions,Aggregate,compute_stats
import datetime,math,json,os,logging
from collections import defaultdict
from ..entity import enums,pvisit_pb2,visit
from ..entity import stream_pb
from ..entity.aggregate import sanitize,process_exclusions
from collections import defaultdict

AGE_MIN = 18
N4_POLICY = aggregate.Policy(min_count=10,min_hospital=2,min_subset=50,base=1)


class N4Coder(codes.Coder):
    def __init__(self):
        super(N4Coder, self).__init__()

    def __getitem__(self, item):
        return super(N4Coder,self).__getitem__(item)


class Node(object):

    def __init__(self):
        self.obj = NEntry()

    def init_compute(self,code,dataset,policy):
        self.key = code
        self.dataset = dataset
        self.policy = policy
        self.count = 0
        self.linked_visits = 0
        self.unlinked_visits = 0
        self.patients = 0

    def add(self,p):
        self.count += 1
        if p.patient_key.startswith('-') or p.patient_key.strip() == "":
            self.unlinked_visits += 1
        else:
            self.patients += 1
            self.linked_visits += len(p.visits)

    def end_compute(self):
        if self.count > self.policy.min_subset:
            self.obj.count = sanitize(self.count,self.policy)
            self.obj.linked_visits = sanitize(self.linked_visits,self.policy)
            self.obj.unlinked_visits = sanitize(self.unlinked_visits,self.policy)
            self.obj.patients = sanitize(self.patients,self.policy)
            return True
        return False


class Edge(object):
    def __init__(self):
        self.obj = NEntry()
        self.patient_keys = set()

    def init_compute(self,e,policy,dataset):
        self.policy = policy
        self.obj.index,self.obj.sub = e
        self.index,self.sub = self.obj.index,self.obj.sub
        self.obj.key = "{}_{}".format(self.obj.index,self.obj.sub)
        self.key = self.obj.key
        self.count = 0
        self.linked_visits = 0
        self.unlinked_visits = 0
        self.patients = 0
        self.index_visits = 0
        self.sub_visits = 0
        self.intersection_visits = 0
        self.patient_index_first = 0
        self.patient_sub_first = 0
        self.index_first_days = defaultdict(int)
        self.sub_first_days = defaultdict(int)
        self.first_inpatient_visit = Aggregate()
        self.first_inpatient_visit.init_compute(self.obj.key,dataset,policy)
        self.counter = defaultdict(int)
        self.type_count = defaultdict(int)
        self.visit_count_histogram = defaultdict(int)

    def db_entry(self,path,exclusions):
        return {'index':self.index,
                'sub':self.sub,
                'count':self.obj.count,
                'patients':self.obj.patients,
                'linked_visits':self.obj.linked_visits,
                'index_visits':self.obj.index_visits,
                'sub_visits':self.obj.sub_visits,
                'intersection_visits':self.obj.intersection_visits,
                'unlinked_visits':self.obj.unlinked_visits,
                'patient_index_first':self.obj.patient_index_first,
                'patient_first':self.obj.patient_index_first,
                'patient_sub_first':self.obj.patient_sub_first,
                'index_first_days':self.obj.index_first_days.median,
                'sub_first_days':self.obj.sub_first_days.median,
                'key':self.obj.key,
                'path':path,
                'exclusions':process_exclusions(exclusions,N4_POLICY)}

    def age_plot(self):
        age_plot_data = { k:0 for k in range(20,100)}
        age_plot_data.update({t.k:t.v for t in self.obj.first_inpatient_visit.ageh.h})
        return age_plot_data.items()

    def add_codes(self,p):
        tuple_set = set()
        for v in p.visits:
            for field,codes in [('exh',v.exs),('dx',v.dxs),('dx_poa',v.poas),('dx_prim',[v.primary_diagnosis,]),('drgh',[v.drg,])]:
                for code in codes:
                    tuple_set.add((field,code))
            for field,codes in [('primary_prh',[v.primary_procedure]),('prh',v.prs)]:
                for code in codes:
                    tuple_set.add((field,code.pcode))
        for t in tuple_set:
            self.counter[t] += 1

    def add_vtypes(self,p):
        self.visit_count_histogram[len(p.visits)] += 1
        for v in p.visits:
            self.type_count[('All',v.vtype)] += 1
            if self.index in v.dxs:
                self.index_visits += 1
                self.type_count[(self.index,v.vtype)] += 1
            if self.sub in v.dxs:
                self.sub_visits += 1
                self.type_count[(self.sub,v.vtype)] += 1
                if self.index in v.dxs:
                    self.intersection_visits += 1
                    self.type_count[('Both',v.vtype)] += 1

    def add_vdays(self,p):
        index_days = [v.day for v in p.visits if self.index in v.dxs and v.day > 0]
        sub_days = [v.day for v in p.visits if self.sub in v.dxs and v.day > 0]
        if index_days and sub_days:
            if min(index_days) < min(sub_days):
                self.patient_index_first += 1
                self.index_first_days[min(sub_days) - min(index_days)] +=1
            elif min(sub_days) < min(index_days):
                self.patient_sub_first += 1
                self.sub_first_days[(min(index_days) - min(sub_days))] += 1

    def add_first_inpatient(self,p):
        for v in p.visits:
            if v.vtype == enums.IP:
                self.first_inpatient_visit.add(v)
                break

    def add(self,p):
        self.count += 1
        if p.patient_key.startswith('-') or p.patient_key.strip() == "":
            self.unlinked_visits += 1
        else:
            self.patients += 1
            self.linked_visits += len(p.visits)
            self.add_codes(p)
            self.add_vtypes(p)
            self.add_vdays(p)
            self.add_first_inpatient(p)

    def end_compute(self):
        if self.count > self.policy.min_subset:
            if self.first_inpatient_visit.end_compute():
                self.obj.first_inpatient_visit.CopyFrom(self.first_inpatient_visit.obj)
            self.obj.count = sanitize(self.count,self.policy)
            self.obj.linked_visits = sanitize(self.linked_visits,self.policy)
            self.obj.unlinked_visits = sanitize(self.unlinked_visits,self.policy)
            self.obj.patients = sanitize(self.patients,self.policy)
            self.obj.index_visits = sanitize(self.index_visits,self.policy)
            self.obj.sub_visits = sanitize(self.sub_visits,self.policy)
            self.obj.intersection_visits = sanitize(self.intersection_visits,self.policy)
            self.obj.patient_index_first = sanitize(self.patient_index_first,self.policy)
            self.obj.patient_sub_first = sanitize(self.patient_sub_first,self.policy)
            self.end_compute_codes()
            if self.index_first_days.items() and len(self.index_first_days) > 10:
                mean,median,fq,tq = compute_stats(self.index_first_days)
                self.obj.index_first_days.median = int(round(median))
                self.obj.index_first_days.mean = round(mean,2)
                self.obj.index_first_days.fq = int(round(fq))
                self.obj.index_first_days.tq = int(round(tq))
            else:
                self.obj.index_first_days.median = 0
                self.obj.index_first_days.mean = 0
                self.obj.index_first_days.fq = 0
                self.obj.index_first_days.tq = 0
            if self.sub_first_days.items() and len(self.sub_first_days) > 10:
                mean,median,fq,tq = compute_stats(self.sub_first_days)
                self.obj.sub_first_days.median = int(round(median))
                self.obj.sub_first_days.mean = round(mean,2)
                self.obj.sub_first_days.fq = int(round(fq))
                self.obj.sub_first_days.tq = int(round(tq))
            else:
                self.obj.sub_first_days.median = 0
                self.obj.sub_first_days.mean = 0
                self.obj.sub_first_days.fq = 0
                self.obj.sub_first_days.tq = 0
            temp = {}
            for k in ['All','Both',self.index,self.sub]:
                temp[k] = self.obj.vtype_count.add()
                temp[k].k = k
            for k,vtype in self.type_count:
                temp[k].k = k
                if vtype == enums.IP:
                    temp[k].IP = int(self.policy.base*int(math.floor(self.type_count[(k,vtype)]/float(self.policy.base)))) if self.type_count[(k,vtype)] > self.policy.min_count else 0
                elif vtype == enums.ED:
                    temp[k].ED = int(self.policy.base*int(math.floor(self.type_count[(k,vtype)]/float(self.policy.base)))) if self.type_count[(k,vtype)] > self.policy.min_count else 0
                elif vtype == enums.AS:
                    temp[k].AS = int(self.policy.base*int(math.floor(self.type_count[(k,vtype)]/float(self.policy.base)))) if self.type_count[(k,vtype)] > self.policy.min_count else 0
                else:
                    raise ValueError
            return True
        return False

    def end_compute_codes(self):
        combined_dx = defaultdict(lambda :{'primary':0,'poa':0,'all':0})
        for k,v in self.counter.iteritems():
            if v > self.policy.min_count:
                if not k[0].startswith('dx') and k[0] not in Aggregate.int_types:
                    temp = self.obj.__getattribute__(k[0]).add()
                    temp.k = k[1]
                    temp.v = int(self.policy.base*int(math.floor(v/float(self.policy.base)))) if v > self.policy.min_count else 0
                elif k[0] == 'dx_prim':
                    combined_dx[k[1]]['primary'] = int(self.policy.base*int(math.floor(v/float(self.policy.base)))) if v > self.policy.min_count else 0
                elif k[0] == 'dx_poa':
                    combined_dx[k[1]]['poa'] = int(self.policy.base*int(math.floor(v/float(self.policy.base)))) if v > self.policy.min_count else 0
                elif k[0] == 'dx':
                    combined_dx[k[1]]['all'] = int(self.policy.base*int(math.floor(v/float(self.policy.base)))) if v > self.policy.min_count else 0
        for k,v in combined_dx.iteritems():
            temp = self.obj.dxh.add()
            temp.k = k
            temp.primary = v['primary']
            temp.poa = v['poa']
            temp.all = v['all']


    def SerializeToString(self):
        return self.obj.SerializeToString()

    def ParseFromString(self,s):
        self.obj.ParseFromString(s)



def compute_relative_risk(N,P1,P2,C12):
    rr = (float(C12)*N)/(float(P1)*float(P2))
    correlation =  ((float(C12)*N)-(float(P1)*float(P2)))/math.sqrt((N-P1)*(N-P2)*float(P1)*float(P2))
    return rr,correlation


def aggregate_patients(code,patients,dataset,result_dir,total_patients,reduce_mode_mini=True):
    try:
        os.makedirs(result_dir)
    except OSError:
        pass
    logging.info("Start N4 {}, reduce mode {} ".format(code,reduce_mode_mini))
    edges,index_edges,total_patients,index_patients,edge_exclusion_counters = process_node(code,dataset,patients,result_dir,total_patients)
    logging.info("processed Node, starting edges {}".format(len(edges)))
    edges_result = process_edge(edges,index_edges,edge_exclusion_counters,patients,result_dir)
    updates = []
    for e,fname,exclusions in edges_result:
        updates.append({'index': e.index,
                        'sub': e.sub,
                        'filename': fname,
                        'exclusions': process_exclusions(exclusions,N4_POLICY),
                        'patients': e.patients,
                        'key': e.key})
    logging.info("Finished N4 {}, reduce mode {} ".format(code,reduce_mode_mini))
    result = {'updates': updates, 'code': code}
    with open("{}/{}.patients_meta.json".format(result_dir, code), 'w') as fh:
        json.dump(result, fh)
    return result


def defaultdict_int():
    return defaultdict(int)


def process_node(code,dataset,patients,fout,total_patients,code_sub=None):
    exclusion_counter = defaultdict(int)
    edge_exclusion_counters = defaultdict(defaultdict_int)
    edges = {}
    node = Node()
    node.init_compute(code,dataset,N4_POLICY)
    index_edge = defaultdict(set)
    i = 0
    for i,p in enumerate(patients):
        if check_criteria_node(p,exclusion_counter):
            node.add(p)
        edge_set = set()
        for v in p.visits:
            for dx in v.dxs:
                if (code_sub is None) or (code_sub == dx):
                    e = (code,dx)
                    edge_set.add(e)
                    if e not in edges:
                        edges[e] = Edge()
                        edges[e].init_compute(e,N4_POLICY,dataset)
        for e in edge_set:
            if check_criteria_edge(p,edge_exclusion_counters[e]):
                index_edge[i].add(e)
    index_patients = node.patients
    return edges,index_edge,total_patients,index_patients,edge_exclusion_counters


def process_edge(edges,index_edge,edge_exclusion_counters,patients,fout):
    for i,p in enumerate(patients):
        for e in index_edge[i]:
            edges[e].add(p)
    edges_result = []
    for edge in edges.itervalues():
        if edge.end_compute():
            fname = "{}.{}".format(edge.obj.key,"n4_edge")
            foutput = open("{}/{}".format(fout,fname),'w')
            foutput.write(edge.SerializeToString())
            foutput.close()
            edges_result.append((edge,fname,edge_exclusion_counters[(edge.obj.index,edge.obj.sub)]))
    return edges_result


def check_criteria_node(p,counter):
    min_age = min([v.age for v in p.visits])
    counter['total'] += 1
    if p.patient_key.startswith('-') or p.patient_key.strip() == "":
        counter["excluded"] += 1
        counter[(1,"unlinked visit")] += 1
        return False
    elif min_age <= AGE_MIN:
        counter['excluded'] += 1
        counter[(2,"Age < {}".format(AGE_MIN))] += 1
        return False
    else:
        counter['selected'] += 1
        return True


def check_criteria_edge(p,counter):
    min_age = min([v.age for v in p.visits])
    counter['total'] += 1
    if p.patient_key.startswith('-') or p.patient_key.strip() == "":
        counter["excluded"] += 1
        counter[(1,"unlinked visit")] += 1
        return False
    elif min_age <= AGE_MIN:
        counter['excluded'] += 1
        counter[(2,"Age < {}".format(AGE_MIN))] += 1
        return False
    else:
        counter['selected'] += 1
        return True
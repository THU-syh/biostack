import sys,gzip,base64,urllib
sys.path.append('../')
import logging,json
import chlib,pickle,os
from chlib.entity.enums import ALIVE,DEAD,D_HOSPITAL
from chlib.entity.pvisit_pb2 import Patient
from chlib.entity.pml_pb2 import PDXCLASSIFIER
Coder = chlib.codes.Coder()
try:
    import numpy as np
    from scipy.sparse import lil_matrix
    from sklearn import preprocessing
    from collections import defaultdict
    from .features import VisitToFeature
    from sklearn.ensemble import RandomForestClassifier
    from scipy import interp
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    import matplotlib.pyplot as plt
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
except ImportError:
    pass

EXCLUDED = {'DV3000','DV3001','D64511','D66411','D66401','D64891','D65971','D650','D27801','D29181','D65811','D65811','D66331','D64421','D65961','D29633'}


def get_terminal_or_edge(p,index_code,exclusions,exclude_year,exclude_quarter):
    skipped = False
    for i, v in enumerate(p.visits):
        if v.primary_diagnosis == index_code and v.age < 0:
            exclusions["Excluding patient due to age < 0"] += 1
            return None,None,None
        elif v.primary_diagnosis == index_code and v.death != ALIVE:
            exclusions["Excluding patient due to death during index visit"] += 1
            return None,None,None
        elif v.primary_diagnosis == index_code:
            if i + 1 < len(p.visits):
                sub = p.visits[i + 1]
                delta = sub.day - (v.day + v.los)
                if delta < 0:
                    exclusions['Excluding patient due to days between revisit < 0'] += 1
                    return None,None,None
                if v.disposition != D_HOSPITAL:
                    return v, sub, delta # return the very first rehospitalization
                else:
                    skipped = True  # This one lead to a transfer
            else:
                if v.primary_diagnosis == index_code and v.year == exclude_year and v.quarter == exclude_quarter:
                    exclusions["Excluding patient, terminal visit during {} Quarter {}".format(exclude_year,exclude_quarter)] += 1
                    return None, None, None
                else:
                    return v, None, None  # return the single visit
    if skipped:
        exclusions['Excluding patient rehospitalization involved transfer, no other qualifying index visit found'] += 1
    else:
        exclusions['Excluding patient no qualifying index visit found'] += 1
    return None,None,None

class PrimaryDiagnosisReadmission(object):
    
    def __init__(self,code,dataset,visits_count_primary):
        self.dataset = dataset
        self.index_code = code
        self.stats = dataset.get_code(code)
        self.mean_sample_age = None
        self.mean_sample_visits_per_patient = None
        self.subsequent_visits = chlib.entity.aggregate.Aggregate()
        policy = chlib.entity.aggregate.Policy(base=10, min_count=20)
        self.PA = chlib.entity.aggregate.PatientAggregate()
        self.subsequent_visits.init_compute('{}_sub_visit'.format(self.index_code),self.dataset.identifier,policy=policy)
        self.index_visits = chlib.entity.aggregate.Aggregate()
        self.PA.init_compute(key=self.index_code,dataset=self.dataset.identifier,policy=policy)
        self.index_visits.init_compute(key=self.index_code,dataset=self.dataset.identifier,policy=policy)
        self.visits_count_primary = visits_count_primary
        self.label_distribution = defaultdict(int)
        self.features = {}
        self.past_codes = {}
        self.labels = defaultdict(set)
        self.exclusions = defaultdict(int)
        self.aggregate_stats = PDXCLASSIFIER()
        self.visit_to_features = VisitToFeature(dataset=self.dataset)
        self.matrix_index_to_visit = {}
        self.classifiers = {}
        self.count = 0
        for dirname in ["{}/patients/".format(self.dataset.ml_dir),
                        "{}/plots/".format(self.dataset.ml_dir),
                        "{}/models/".format(self.dataset.ml_dir),
                        "{}/stats/".format(self.dataset.ml_dir),
                        "{}/data/".format(self.dataset.ml_dir)]:
            if not os.path.isdir(dirname):
                os.mkdir(dirname)


    def subset(self):
        pkeys, vkeys = self.dataset.get_patient_keys_by_primary_diagnosis(self.index_code)
        pkeys = set(pkeys)
        fname = "{}/patients/{}.pickle".format(self.dataset.ml_dir,self.index_code)
        if not os.path.isfile(fname):
            patients = []
            total = 0
            selected = 0
            print "starting {} with {}".format(self.index_code,len(pkeys))
            for pkey, p in self.dataset.iter_patients_by_code(self.index_code):
                total += 1
                if 'NRD'+pkey in pkeys:
                    selected += 1
                    patients.append(p.SerializeToString())
            print fname,total,selected
            with gzip.open(fname,'w') as fh:
                pickle.dump(patients,fh)


    def iter_patients(self):
        if not os.path.isfile("{}/patients/{}.pickle".format(self.dataset.ml_dir,self.index_code)):
            self.subset()
        with gzip.open("{}/patients/{}.pickle".format(self.dataset.ml_dir,self.index_code)) as fh:
            patients = pickle.load(fh)
        for ps in patients:
            p = Patient()
            p.ParseFromString(ps)
            yield p

    def compute(self):
        for p in self.iter_patients():
            self.count += 1
            v, sub, delta = get_terminal_or_edge(p, self.index_code, self.exclusions,
                                                 exclude_year=max(self.dataset.years),
                                                 exclude_quarter=4)
            if v:
                self.PA.add_patient(p)
                self.features[v.key] = self.visit_to_features.get_features(v)
                self.past_codes[v.key] = self.visit_to_features.all_past_codes(p, v)
                self.index_visits.add(v)
                if sub:
                    if v.key == sub.key:
                        raise ValueError, "Logic error"
                    self.subsequent_visits.add(sub)
                    for delta_thresh in [7, 30]:
                        if delta <= delta_thresh:
                            self.labels[v.key].add('Readmit within {} days'.format(delta_thresh))
                    if delta > 30:
                        self.labels[v.key].add('Readmit after 30 days')
                    for pr in sub.prs:
                        self.labels[v.key].add(pr.pcode)
                    for ex in sub.exs:
                        self.labels[v.key].add(ex)
                    if sub.primary_diagnosis:
                        self.labels[v.key].add(sub.primary_diagnosis)
                    for l in self.labels[v.key]:
                        self.label_distribution[l] += 1

        self.data = {
            'features':self.features,
            'past_codes': self.past_codes,
            'labels':{k:list(v) for k,v in self.labels.iteritems()},
        }
        with open('{}/data/{}.json'.format(self.dataset.ml_dir,self.index_code),'w') as fh:
            json.dump(self.data,fh)
        with open("{}/stats/{}.stats".format(self.dataset.ml_dir, self.index_code), 'w') as fh:
            self.aggregate_stats.count = self.count
            self.aggregate_stats.index_code = self.index_code
            if self.subsequent_visits.end_compute():
                self.aggregate_stats.sub.CopyFrom(self.subsequent_visits.obj)
            if self.index_visits.end_compute():
                self.aggregate_stats.index.CopyFrom(self.index_visits.obj)
            if self.PA.end_compute():
                self.aggregate_stats.patients.CopyFrom(self.PA.obj)
            for k,v in self.exclusions.iteritems():
                temp = self.aggregate_stats.exclusions.add()
                temp.reason = k
                temp.count = v if v > 20 else 0
            for k,v in self.label_distribution.iteritems():
                temp = self.aggregate_stats.labels.add()
                temp.label = k
                temp.count = v if v > 20 else 0
            fh.write(self.aggregate_stats.SerializeToString())
        self.create_XY()

    def load(self):
        fname = "{}/stats/{}.stats".format(self.dataset.ml_dir, self.index_code)
        self.aggregate_stats.ParseFromString(file(fname).read())
        self.count = self.aggregate_stats.count
        fh_data = file('{}/data/{}.json'.format(self.dataset.ml_dir, self.index_code))
        data = json.load(fh_data)
        self.features = data['features']
        self.past_codes = data['past_codes']
        for k,v in data['labels'].iteritems():
            self.labels[k] = set(v)
        for ls in self.labels.itervalues():
            for l in ls:
                self.label_distribution[l] += 1
        self.create_XY()

    def create_XY(self):
        self.X = lil_matrix((len(self.features),self.visit_to_features.current_index))
        self.Y = {}
        candidate_labels = { k for k,v in self.label_distribution.iteritems() if v > 100}
        for l in candidate_labels:
            self.Y[l] = []
        for matrix_index,k in enumerate(self.features):
            for featname,index in self.features[k]:
                self.X[matrix_index,index] = 1
            self.matrix_index_to_visit[matrix_index] = k
            for l in candidate_labels:
                if l in self.labels[k]:
                    self.Y[l].append(1)
                else:
                   self.Y[l].append(0)
        for k,v in self.Y.iteritems():
            self.Y[k] = np.array(v)

    # def get_similar(self,patient_feat,k=30):
    #     return {}

    def add_past_codes(self):
        new_features = 0
        patients_with_past_visits = 0
        for matrix_index,visit_key in self.matrix_index_to_visit.iteritems():
            if visit_key in self.past_codes:
                if self.past_codes[visit_key]:
                    patients_with_past_visits += 1
                for featname, index in self.past_codes[visit_key]:
                    if self.X[matrix_index, index] == 0: # check if new feature  was added
                        new_features += 1
                        self.X[matrix_index, index] = 1
        print new_features,len(self.features),patients_with_past_visits

    def train_models(self,min_count=500):
        for k,v in self.label_distribution.iteritems():
            if v > min_count:
                logging.info("training {}".format(k))
                self.train_random_forest(label=k)
                logging.info("trained {}".format(k))

    def train_random_forest(self,label,n_estimators=100,feature_selection=False):
        if label in self.Y:
            X = self.X
            y = self.Y[label]
            cv = StratifiedKFold(n_splits=4)
            classifier = RandomForestClassifier(n_estimators=n_estimators)
            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
            lw = 2
            i = 0
            for (train, test), color in zip(cv.split(X, y), colors):
                if feature_selection:
                    selector = SelectKBest(chi2, k=400)
                    X_train = selector.fit_transform(X[train], y[train])
                    X_test = selector.transform(X[test])
                else:
                    X_train = X[train]
                    X_test = X[test]
                probas_ = classifier.fit(X_train, y[train]).predict_proba(X_test)
                # Compute ROC curve and area the curve
                fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
                i += 1
            plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k')
            plt.savefig('{}/plots/AUC_{}_{}_RF.png'.format(self.dataset.ml_dir,self.index_code,label.replace(' ','_')))
            plt.clf()
            mean_tpr /= cv.get_n_splits(X, y)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            self.classifiers[label] = classifier
            importances = classifier.feature_importances_
            std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
            indices = np.argsort(importances)[::-1]
            top_20_features = []
            for f in range(X.shape[1])[:20]:
                top_20_features.append((f + 1, self.visit_to_features.index_to_feature[indices[f]],importances[indices[f]], std[indices[f]]))
            fname = '{}/models/{}_label_{}_RF.json'.format(self.dataset.ml_dir,self.index_code,label)
            with open(fname,'w') as fh:
                json.dump({'auc':mean_auc,'top_features':top_20_features},fh)
            fname = '{}/models/{}_label_{}_RF.pkl'.format(self.dataset.ml_dir,self.index_code,label)
            with open(fname,'w') as fh:
                pickle.dump(classifier,fh)
            return mean_auc,top_20_features
        else:
            raise ValueError

    @classmethod
    def list_of_primary_diagnosis(cls,dataset,min_visits=5000):
        primary_diagnoses = []
        for code in dataset.iter_codes():
            if code.code_type == 'dx' and code.visits_count_primary() >= min_visits:
                primary_diagnoses.append(cls(code.code,dataset,code.visits_count_primary()))
        return primary_diagnoses
    
    @classmethod
    def get(cls,dx_code,dataset):
        for code in dataset.iter_codes():
            if code.code_type == 'dx' and code.code == dx_code:
                return cls(code.code,dataset,code.visits_count_primary())
        raise ValueError,"{} Not found".format(dx_code)

    @classmethod
    def create_list(cls,dataset,min_visits=5000):
        dx_list,selected_list = [],[]
        primary_diagnoses = cls.list_of_primary_diagnosis(dataset,min_visits)
        for pdx in primary_diagnoses:
            dx_list.append({
                'dx': pdx.index_code,
                'description': Coder[pdx.index_code],
                'primary_count_total': pdx.visits_count_primary,
                'excluded': pdx.index_code in EXCLUDED
            })
        with open("{}/list.json".format(dataset.ml_dir), 'w') as fh:
            json.dump(dx_list, fh)
        for pdx in primary_diagnoses[:50]:
            if pdx.index_code not in EXCLUDED:
                logging.info("computing stats {}".format(pdx.index_code))
                pdx.subset()
                pdx.compute()
                selected_list.append({
                    'patients': pdx.PA.obj.patient_count,
                    'index_count': pdx.index_visits.obj.count,
                    'sub_count': pdx.subsequent_visits.obj.count,
                    'dx': pdx.index_code,
                    'percent': (100.0 * pdx.subsequent_visits.obj.count) / pdx.index_visits.obj.count,
                    'description': Coder[pdx.index_code],
                    'primary_count_total': pdx.visits_count_primary,
                })
        with open("{}/selected_list.json".format(dataset.ml_dir), 'w') as fh:
            json.dump(selected_list, fh)



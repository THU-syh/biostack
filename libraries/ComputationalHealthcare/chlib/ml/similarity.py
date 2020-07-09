from . import features
import pickle,json
import logging,random,os
from collections import defaultdict,Counter
import hashlib
import math
from ..entity.aggregate import PatientAggregate,Aggregate,Policy,compute_stats
from ..entity.pml_pb2 import NPRESULT
try:
    import numpy as np
    from scipy.sparse import lil_matrix
    from sklearn.neighbors import LSHForest,NearestNeighbors
    from scipy.sparse import csr_matrix,lil_matrix
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


class NearestPatients(object):

    def __init__(self,dataset):
        self.dataset = dataset
        self.visit_to_features = features.VisitToFeature(self.dataset)
        self.exclusions = defaultdict(int)
        self.loaded = False
        self.X = None
        self.matrix_index_to_pkey = {}
        self.nn = LSHForest(random_state=42,min_hash_match=10) # Actually not used
        self.result_path = '{}/np_query_results/'.format(self.dataset.ml_dir)
        if not os.path.isdir(self.result_path):
            os.mkdir(self.result_path)

    def compute_index(self):
        index = {}
        for i,pkey_p in enumerate(self.dataset.iter_patients()):
            if i == 50000:
                break
            if (i+1) % 10000 == 0:
                logging.info("indexed {}".format(i))
            pkey,p = pkey_p
            self.exclusions['total'] += 1
            if p.visits[0].age <= 1:
                self.exclusions["Excluded due to patient younger than 2 years"] += 1
            elif len(p.visits) == 1 and p.visits[0].quarter == 4 and p.visits[0].year == max(self.dataset.years):
                self.exclusions["Excluded due to patient with only one visit. " \
                                "And that visit occurs during fourth quarter of the last year"] += 1
            else:
                self.exclusions["Selected"] += 1
                index[pkey] = list(self.get_vector(p))
        with open('{}/nn_index.pkl'.format(self.dataset.ml_dir),'w') as fh:
            pickle.dump(index,fh)
        with open('{}/nn_stats.json'.format(self.dataset.ml_dir),'w') as fh:
            json.dump({'exclusions':{k:v for k,v in self.exclusions.iteritems()}},fh)
        self.loaded = True

    def load(self):
        logging.info("started loading")
        with open('{}/nn_index.pkl'.format(self.dataset.ml_dir)) as fh:
            index = pickle.load(fh)
        self.X = lil_matrix((len(index),self.visit_to_features.current_index))
        i = 0
        for pkey,v in index.iteritems():
            if v:
                for fid in v:
                    self.X[i,fid] = 1
                self.matrix_index_to_pkey[i] = pkey
                i += 1
            else:
                print pkey,v
                raise ValueError
        index.clear()
        with open('{}/nn_stats.json'.format(self.dataset.ml_dir)) as fh:
            self.exclusions = json.load(fh)['exclusions']
        logging.info("finished loading")
        logging.info("starting fit")
        self.nn.fit(self.X)
        logging.info("finished fit")
        self.loaded = True

    def get_vector(self,p):
        features = []
        if len(p.visits) == 1:
            features += self.visit_to_features.get_features(p.visits[0])
            return set([findex for fname, findex in features])
        else:
            last_visit = p.visits[-1]
            second_last_visit = p.visits[-2]
            if last_visit.day < second_last_visit.day:
                raise ValueError,"Data Error"
            features += self.visit_to_features.get_features(second_last_visit)
            features += self.visit_to_features.all_past_codes(p,second_last_visit)
            return set([findex for fname,findex in features])

    def find_k_nearest_match(self,vwdict,k_neighbors=1000):
        """
        Really slow and stupid implementation of matching distance since the sci-kit learn does not supports it with sparse arrays.
        :param vlist:
        :param k:
        :return:
        """
        if not self.loaded:
            self.load()
        logging.info("starting query")
        results = []
        v = []
        for fname,fweight in vwdict.iteritems():
            if fname in self.visit_to_features.feature_to_index:
                v.append((self.visit_to_features.feature_to_index[fname],fweight))
        for i in range(len(self.matrix_index_to_pkey)):
            results.append((sum([w for k,w in v if self.X[i,k] == 1]),i))
        results.sort(reverse=True)
        policy = Policy()
        indices = [i for c,i in results[:k_neighbors]]
        patient_keys = [self.matrix_index_to_pkey[i] for i in indices]
        logging.info("finished query")
        logging.info("getting patients")
        patients = [self.dataset.get_patient(pkey)[1] for pkey in patient_keys]
        patient_stats = PatientAggregate()
        aggregate_results = NPRESULT()
        patient_stats.init_compute("Patients with last visit excluded",dataset=self.dataset.identifier, policy=policy)
        future_visits = Aggregate(mini=False)
        future_visits.init_compute("Future visits", dataset=self.dataset.identifier,policy=policy)
        index_visits = Aggregate(mini=False)
        index_visits.init_compute("Index visit", dataset=self.dataset.identifier,policy=policy)
        delta_counter = defaultdict(int)
        logging.info("computing aggregate statistics")
        for p in patients:
            if len(p.visits) > 1:
                index = p.visits[-2]
                future = p.visits[-1]
                patient_stats.add_patient(p)
                future_visits.add(future)
                index_visits.add(index)
                delta = future.day - ( index.day + index.los )
                if delta >= 0:
                    delta_counter[delta] += 1
            else:
                index = p.visits[0]
                patient_stats.add_patient(p)
                index_visits.add(index)
        query_id = hashlib.sha256(repr(sorted(vwdict.items()))).hexdigest()
        models = NeighborhoodModels(query_id=query_id,dataset=self.dataset)
        models.compute(patients)
        models.train_models()
        patient_stats.end_compute()
        index_visits.end_compute()
        logging.info("finished processing query")
        aggregate_results.neighbors = k_neighbors
        aggregate_results.patients.CopyFrom(patient_stats.obj)
        aggregate_results.index.CopyFrom(index_visits.obj)
        if future_visits.end_compute():
            aggregate_results.future.CopyFrom(future_visits.obj)
        for k,v in vwdict.iteritems():
            temp = aggregate_results.query.add()
            temp.fname = k
            temp.weight = v
        mean, median, fq, tq = compute_stats(delta_counter)
        aggregate_results.deltah.median = int(round(median))
        aggregate_results.deltah.mean = round(mean, 2)
        aggregate_results.deltah.fq = int(round(fq))
        aggregate_results.deltah.tq = int(round(tq))
        for value, c in delta_counter.iteritems():
            if value >= 0:
                temp = aggregate_results.deltah.h.add()
                temp.k = value
                temp.v = int(policy.base * int(math.floor(c / float(policy.base)))) if c > policy.min_count else 0
        with open('{}/{}.npresults'.format(self.result_path,query_id),'w') as fh:
            fh.write(aggregate_results.SerializeToString())
        return aggregate_results, patient_stats, future_visits, index_visits

    def get_result(self):
        pass



class NeighborhoodModels(object):
    def __init__(self,query_id, dataset):
        self.dataset = dataset
        self.query_id = query_id
        self.label_distribution = defaultdict(int)
        self.features = {}
        self.past_codes = {}
        self.labels = defaultdict(set)
        self.exclusions = defaultdict(int)
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

    def compute(self,patients):
        for p in patients:
            self.count += 1
            v,sub,delta = None, None ,None
            if len(p.visits) == 1:
                v = p.visits[0]
                sub = None
                delta = None
            else:
                v = p.visits[-2]
                sub = p.visits[-1]
                delta = sub.day - (v.day + v.los)
                if delta < 0 or delta > 90:
                    sub = None
            if v:
                self.features[v.key] = self.visit_to_features.get_features(v)
                self.past_codes[v.key] = self.visit_to_features.all_past_codes(p, v)
                if sub:
                    if v.key == sub.key:
                        raise ValueError, "Logic error"
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
            'features': self.features,
            'past_codes': self.past_codes,
            'labels': {k: list(v) for k, v in self.labels.iteritems()},
        }
        with open('{}/data/{}.json'.format(self.dataset.ml_dir, self.query_id), 'w') as fh:
            json.dump(self.data, fh)
        self.create_XY()

    def create_XY(self):
        self.X = lil_matrix((len(self.features), self.visit_to_features.current_index))
        self.Y = {}
        candidate_labels = {k for k, v in self.label_distribution.iteritems() if v > 50}
        logging.info(candidate_labels)
        for l in candidate_labels:
            self.Y[l] = []
        for matrix_index, k in enumerate(self.features):
            for featname, index in self.features[k]:
                self.X[matrix_index, index] = 1
            self.matrix_index_to_visit[matrix_index] = k
            for l in candidate_labels:
                if l in self.labels[k]:
                    self.Y[l].append(1)
                else:
                    self.Y[l].append(0)
        for k, v in self.Y.iteritems():
            self.Y[k] = np.array(v)

    def add_past_codes(self):
        new_features = 0
        patients_with_past_visits = 0
        for matrix_index, visit_key in self.matrix_index_to_visit.iteritems():
            if visit_key in self.past_codes:
                if self.past_codes[visit_key]:
                    patients_with_past_visits += 1
                for featname, index in self.past_codes[visit_key]:
                    if self.X[matrix_index, index] == 0:  # check if new feature  was added
                        new_features += 1
                        self.X[matrix_index, index] = 1
        print new_features, len(self.features), patients_with_past_visits

    def train_models(self, min_count=50):
        for k, v in self.label_distribution.iteritems():
            if v > min_count:
                logging.info("training {}".format(k))
                self.train_random_forest(label=k)
                logging.info("trained {}".format(k))

    def train_random_forest(self, label, n_estimators=100, feature_selection=False):
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
            plt.savefig(
                '{}/plots/AUC_{}_{}_RF.png'.format(self.dataset.ml_dir, self.query_id, label.replace(' ', '_')))
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
                top_20_features.append((f + 1, self.visit_to_features.index_to_feature[indices[f]],
                                        importances[indices[f]], std[indices[f]]))
            fname = '{}/models/{}_label_{}_RF.json'.format(self.dataset.ml_dir, self.query_id, label)
            with open(fname, 'w') as fh:
                json.dump({'auc': mean_auc, 'top_features': top_20_features}, fh)
            fname = '{}/models/{}_label_{}_RF.pkl'.format(self.dataset.ml_dir, self.query_id, label)
            with open(fname, 'w') as fh:
                pickle.dump(classifier, fh)
            return mean_auc, top_20_features
        else:
            raise ValueError




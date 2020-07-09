from __future__ import absolute_import
from celery import shared_task
import chlib,base64,logging
from chlib.ml import classifiers

try:
    import plyvel
except ImportError:
    logging.warning("could not import plyvel")
    plyvel = None

DATASETS = {}
DB,DBS,DB_EXAMPLES = {},{},{}
Q_DATA = 'qdata'
CLASSIFER_MODELS = {}

def load_db():
    DATASETS.update(chlib.data.Data.load_from_config('config.json'))
    for k,v in DATASETS.iteritems():
        try:
            DB[k] = plyvel.DB(v.db, create_if_missing=False)
            DBS[k] = DB[k].snapshot()
            for example,example_v in DBS[k]:
                DB_EXAMPLES[k] = example
                break
        except:
            logging.exception("could not load {}".format(k))
            pass


def close_db():
    for v in DB.itervalues():
        v.close()
    DB.clear()
    DBS.clear()

@shared_task
def data_examples():
    if len(DB) == 0:
        load_db()
    return DB_EXAMPLES


@shared_task
def data_get_patient(db,key):
    if len(DB) == 0:
        load_db()
    v = DBS[db].get(key.encode("utf-8"))
    next_key = None
    for k, v in DBS[db].iterator(start=key.encode('utf-8'),include_start=False):
        next_key = k
        break
    if v:
        coded = base64.encodestring(v)
    else:
        coded = ""
    return {'key':key.encode("utf-8"),'data':coded,'next':next_key.encode("utf-8")}


@shared_task
def get_similar_patient(dataset,primary_diangosis,patient_features):
    pass

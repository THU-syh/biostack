import os,sys,logging,shutil,random,time,boto3,glob,gzip,json,datetime
from collections import defaultdict
import chlib
import django,plyvel
from fabric.api import env,local,run,sudo,put,cd,lcd,puts,task


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/fab.log',
                    filemode='a')


@task
def compile_protocols():
    protocols = ["penums.proto","pvisit.proto","pstat.proto","pn1.proto","pn2.proto","pn3.proto","pn4.proto"]
    with lcd('chlib/entity/protocols'):
        for fname in protocols:
            local('protoc --python_out=../ --cpp_out=../../cppcode/protos/ {}'.format(fname))




@task
def clear_logs():
    """
    remove logs
    """
    local('rm logs/*.log &')
    local('rm logs/cpp* &')


@task
def start_server():
    import time
    time.sleep(60) # wait for other containers to come online
    local("python manage.py makemigrations")
    local("python manage.py migrate")
    local("python manage.py runserver 0.0.0.0:8000")




@task
def process_code(code):
    datasets = chlib.data.Data.load_from_config('config.json')
    for k,d in datasets.iteritems():
        logging.info("started {}".format(d.identifier))
        try:
            d.process_code(code)
        except plyvel._plyvel.Error:
            pass
            logging.info("could not open database for {}".format(d.identifier))
        else:
            d.close_db()
            logging.info("finished {}".format(d.identifier))


@task
def prepare_tx(skip_prepare=False):
    TX = chlib.data.Data.get_from_config('config.json','TX')
    TX.setup(skip_prepare,False,False,test=False)


@task
def prepare_nrd(skip_prepare=False):
    HCUPNRD = chlib.data.Data.get_from_config('config.json','HCUPNRD')
    HCUPNRD.setup(skip_prepare,False,False,test=False)
    compute_nn()

@task
def precompute(dataset):
    D = chlib.data.Data.get_from_config('config.json',dataset)
    D.pre_compute()


def process_code_dataset(code_dataset_id):
    code,dataset_id = code_dataset_id
    logging.info("started {}".format(code))
    dataset = chlib.data.Data.get_from_config('config.json',dataset_id)
    dataset.process_code(code)
    logging.info("finished {}".format(code))


@task
def process_all(dataset_id):
    import multiprocessing
    dataset = chlib.data.Data.get_from_config('config.json', dataset_id)
    pool = multiprocessing.Pool()
    codes = [(c.code, dataset_id) for c in dataset.iter_codes() if c.visits_count() > 100]
    logging.info("starting {} codes for {} ".format(len(codes), dataset_id))
    pool.map(process_code_dataset, codes)
    pool.close()
    logging.info("finished {} codes for {} ".format(len(codes), dataset_id))

@task
def compile_cpp_code():
    with lcd("chlib/cppcode/"):
        local("cmake .")
        local("make")
        try:
            local("mkdir -p bin/Debug/")
        except:
            pass
        local("mv cpp bin/Debug/")


@task
def compute_nn():
    """
    Compute Nearest neighbour index for HCUP NRD Database
    :return:
    """
    from chlib.ml.similarity import NearestPatients
    json_config = 'config.json'
    dataset = chlib.data.Data.get_from_config(json_config_path=json_config, dataset_id='HCUPNRD')
    nn = NearestPatients(dataset)
    nn.compute_index()


@task
def query_nn():
    from chlib.ml.similarity import NearestPatients
    json_config = 'config.json'
    dataset = chlib.data.Data.get_from_config(json_config_path=json_config, dataset_id='HCUPNRD')
    nn = NearestPatients(dataset)
    vweights = {'D486':1,'P9604':1,'P9904':1,'P9904_2':1,'Age_5':1,'D340':4}
    print "using following input"
    print vweights
    ar, pstats, fstats , istats = nn.find_k_nearest_match(vweights,1000)
    print fstats.visualize(host='127.0.0.1',port=8000,prefix='local/')
    print istats.visualize(host='127.0.0.1',port=8000,prefix='local/')
    print pstats.visualize(host='127.0.0.1',port=8000,prefix='local/')


@task
def ci():
    try:
        os.mkdir('test_reports')
    except:
        pass
    now = str(datetime.datetime.utcnow())
    os.mkdir("test_reports/{}".format(now))
    local("fab build && sleep 30 && fab start && sleep 30 && fab test")
    with lcd("test_reports/{}".format(now)):
        local('docker cp $(docker ps -l -q):/root/CH/logs/fab.log .')
    local('fab server')
    with lcd("test_reports/{}".format(now)):
        time.sleep(120)
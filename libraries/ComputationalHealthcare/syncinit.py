import logging,json,glob,os
from CHL.models import SCount,STCount,Code,SYTCount,CodeCount,Dataset,TextSearch
from chlib.codes import Coder
from collections import defaultdict


def init_db(datasets,heroku=False):
    Dataset.objects.all().delete()
    Code.objects.all().delete()
    CodeCount.objects.all().delete()
    STCount.objects.all().delete()
    SCount.objects.all().delete()
    SYTCount.objects.all().delete()
    for k,d in datasets.iteritems():
        init_selected(d,heroku)


def init_selected(d,heroku=False):
    if os.path.isfile(d.base_dir + "/counts.txt"):
        logging.info("starting {}".format(d.identifier))
        dm = Dataset()
        dm.identifier = d.identifier
        dm.linked = d.linked
        dm.base_dir = d.base_dir
        dm.years = d.years
        dm.states = d.states
        dm.patients_count = d.patients
        dm.linked_count = d.linked_visits
        dm.unlinked_count = d.unlinked_visits
        dm.aggregate_patients = d.aggregate_patients
        dm.aggregate_readmits = d.aggregate_readmits
        dm.aggregate_visits = d.aggregate_visits
        dm.aggregate_revisits = d.aggregate_revisits
        dm.name = d.name
        dm.save()
        init_scount(d,dm,heroku)
        init_sytcount(d,dm,heroku)
        init_stcount(d,dm,heroku)
        init_codes(d, dm,heroku)


def init_scount(d,dm,heroku):
    for s in d.states:
        sc = SCount()
        sc.dataset = dm
        sc.state = s
        sc.patients_count = d.state_counts[s]['patients']
        sc.unlinked_count = d.state_counts[s]['unlinked_visits']
        sc.linked_count = d.state_counts[s]['linked_visits']
        sc.save()
    logging.info("finished states")


def init_sytcount(d,dm,heroku):
    for k,v in d.year_counts.iteritems():
        state, vtype, year, linked = k
        syt = SYTCount()
        syt.state = state
        syt.dataset = dm
        syt.visit_type = vtype
        syt.year = year
        syt.linked = True if linked == 'linked' else False
        syt.count = v
        syt.save()
    logging.info("finished SYT")


def init_stcount(d, dm,heroku):
    for k, v in d.type_counts.iteritems():
        state,vtype,linked = k
        st = STCount()
        st.state = state
        st.dataset = dm
        st.visit_type = vtype
        st.linked = True if linked == 'linked' else False
        st.count = v
        st.save()
    logging.info("finished ST")


def init_codes(d,dm,heroku):
    coder = Coder()
    Codes = {}
    for state,year,linked,vtype,ctype,code,count in d.iter_code_counts():
        if ctype != 'pdx' and code not in Codes:
            cd = Code()
            cd.code = code
            cd.description = coder[code]
            cd.code_type = ctype
            cd.dataset = dm
            cd.save()
            Codes[code] = cd
    logging.info("finished Codes")
    objs = [CodeCount(year=year,
                      state=state,
                      linked=linked,
                      visit_type=vtype,
                      count=count,
                      code=code,
                      code_type=ctype,
                      dataset_identifier=dm.identifier)
            for state,year,linked,vtype,ctype,code,count in d.iter_code_counts()]
    logging.info("starting code counts batch")
    CodeCount.objects.bulk_create(objs, batch_size=5000)
    logging.info("finished code counts")


def sync_text():
    desc = defaultdict(str)
    dcount = defaultdict(int)
    ctype = defaultdict(str)
    TextSearch.objects.all().delete()
    for k in Code.objects.all():
        desc[k.code] = k.code + " " + k.description
        dcount[k.code] += 1
        ctype[k.code] = k.code_type
    for k in desc:
        temp = TextSearch()
        temp.code = k
        temp.description = desc[k]
        temp.datasets_count = dcount[k]
        temp.code_type = ctype[k]
        temp.save()
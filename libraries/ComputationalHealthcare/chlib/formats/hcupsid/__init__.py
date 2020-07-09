__author__ = 'aub3'
import re,formats,os,gzip,logging
import merge
from ...entity import enums,visit
PARSERS = {}

DXATADMIT_format = re.compile('DXATADMIT[0-9]{1,2}')
DX_format = re.compile('DX[0-9]{1,2}')
DXPOA_format = re.compile('DXPOA[0-9]{1,2}')
PR_format = re.compile('PR[0-9]{1,2}')
CPT_format = re.compile('CPT[0-9]{1,2}')
EX_format = re.compile('ECODE[0-9]{1,2}')
PRCCS_format = re.compile('CPTCCS[0-9]{1,2}')
f = formats.SASFormat()

def get_zip(d):
    if'MEDINCSTQ' in d:
        return 'MEDINCSTQ'
    elif'ZIPINC_QRTL' in d:
        return 'ZIPINC_QRTL'
    else:
        return None

def process_buffer_hcup(line):
    p = parse(line)
    return p.patient_key,p.visits[0].key,p.SerializeToString()


def finalize_hcup(res,unlinked,state,db):
    for patient_key,visit_key,pstr in res:
        wb = db.write_batch()
        if unlinked:
            wb.put(("U"+state+visit_key).encode("utf-8"),pstr)
        else:
            wb.put((state+patient_key).encode("utf-8"),pstr)
        wb.write()


def parse(line):
    pobj = visit.Patient()
    pobj.raw = line
    if '\t' in line:
        entries = line.strip('\n').split('\t')
        pobj.patient_key = entries[0]
        pobj.linked = True
        for entry in entries[1:]:
            vobj = pobj.visits.add()
            process_entry(entry,vobj)
            visit.index_procedures(vobj)
        visit.sort_visits(pobj)
    else:
        entry = line.strip('\n')
        pobj.patient_key = '-1'
        pobj.linked = False
        vobj = pobj.visits.add()
        process_entry(entry,vobj)
        visit.index_procedures(vobj)
    return pobj





def process_entry(entry,vobj):
    vlink,state,year,ftype,dtyep,days,dline = entry.split('_')
    visit_type = None
    if ftype == 'SID':
        visit_type = 'IP'
        vobj.vtype = enums.IP
    elif ftype == 'SEDD':
        visit_type = 'ED'
        vobj.vtype = enums.ED
    elif ftype == 'SASD':
        visit_type = 'AS'
        vobj.vtype = enums.AS
    ftuple = (state,ftype,int(year),dtyep)
    if not ftuple in PARSERS:
        PARSERS[ftuple] = f.get_format(state,ftype,int(year),dtyep)
    vobj.key = PARSERS[ftuple].get_element('KEY',dline)
    pdx = PARSERS[ftuple].get_element('DX1',dline)
    if pdx.strip():
        vobj.primary_diagnosis = 'D{}'.format(PARSERS[ftuple].get_element('DX1',dline))
    vobj.patient_key = vlink
    vobj.state = state
    vobj.day = int(days)
    vobj.age = int(PARSERS[ftuple].get_element('AGE',dline))
    source = PARSERS[ftuple].get_element('ASOURCE',dline)
    if source and source.strip():
        vobj.source = formats.LMAP[('source_dict',int(source))][2]
    else:
        vobj.source = formats.LMAP[('source_dict',-9)][2]
    vobj.race = formats.LMAP[('race_dict',int(PARSERS[ftuple].get_element('RACE',dline)))][2]
    vobj.sex = formats.LMAP[('sex_dict',int(PARSERS[ftuple].get_element('FEMALE',dline)))][2]
    vobj.payer = formats.LMAP[('payer_dict',int(PARSERS[ftuple].get_element('PAY1',dline)))][2]
    vobj.disposition = formats.LMAP[('disposition_dict',int(PARSERS[ftuple].get_element('DISPUNIFORM',dline)))][2]
    vobj.death = formats.LMAP[('died_dict',int(PARSERS[ftuple].get_element('DIED',dline)))][2]
    vobj.year = int(year)
    month = PARSERS[ftuple].get_element('AMONTH',dline)
    if month:
        vobj.month = int(month)
    else:
        vobj.month = -1
    vobj.quarter = int(PARSERS[ftuple].get_element('DQTR',dline))
    charges = PARSERS[ftuple].get_element('TOTCHG',dline)
    zip_element = get_zip(PARSERS[ftuple].columns)
    if zip_element:
        vobj.zip = formats.LMAP[('pzip_dict',int(PARSERS[ftuple].get_element(zip_element,dline)))][2]
    if charges:
        vobj.charge = float(charges)
    else:
        vobj.charge = -1
    if state == 'CA':
        dnr = PARSERS[ftuple].get_element('DNR',dline)
        if dnr and dnr.strip():
            vobj.dnr = formats.LMAP[('dnr_dict',int(dnr))][2]
        else:
            vobj.dnr = formats.LMAP[('dnr_dict',-9)][2]
    else:
        vobj.dnr = formats.LMAP[('dnr_dict',-9)][2]
    vobj.dataset = state
    if 'DRG24'in PARSERS[ftuple].columns:
        vobj.drg = 'DG{}'.format(PARSERS[ftuple].get_element('DRG24',dline))
    else:
        vobj.drg = 'DG-1'
    los = PARSERS[ftuple].get_element('LOS',dline)
    if los.strip() and los >= 0:
        vobj.los = int(los)
    else:
        vobj.los = -1
    vobj.facility = PARSERS[ftuple].get_element('DSHOSPID',dline)
    dxlist = map(lambda x:'D{}'.format(x),filter(None,[PARSERS[ftuple].get_element(k,dline) for k in PARSERS[ftuple].columns if DX_format.findall(k)]))
    for k in dxlist:
        vobj.dxs.append(k)
    exlist = map(lambda x:'E{}'.format(x),filter(None,[PARSERS[ftuple].get_element(k,dline) for k in PARSERS[ftuple].columns if EX_format.findall(k)]))
    for k in exlist:
        vobj.exs.append(k)
    if visit_type == 'IP':
        primary_procedure = PARSERS[ftuple].get_element('PR1',dline)
        if len(primary_procedure.strip()) > 1:
            vobj.primary_procedure.pcode = 'P'+primary_procedure
            vobj.primary_procedure.ctype = enums.ICD
            vobj.primary_procedure.pday = int(PARSERS[ftuple].get_element('PRDAY1',dline))
        prlist = map(lambda x:(int(x[0]),'P{}'.format(x[1])),filter(lambda x:x[1],[(PARSERS[ftuple].get_element(k.replace('PR','PRDAY'),dline),PARSERS[ftuple].get_element(k,dline)) for k in PARSERS[ftuple].columns if PR_format.findall(k)]))
        for pr in prlist:
            temp = vobj.prs.add()
            temp.pcode = pr[1]
            temp.pday = pr[0]
            temp.ctype = enums.ICD
    elif visit_type == 'ED' or visit_type == 'AS':
        primary_procedure = PARSERS[ftuple].get_element('CPT1',dline)
        if len(primary_procedure.strip()) > 1:
            vobj.primary_procedure.pcode = primary_procedure
            vobj.primary_procedure.ctype = enums.CPT
            vobj.primary_procedure.pday = 0
        prlist = map(lambda x:'C{}'.format(x),filter(None,[PARSERS[ftuple].get_element(k,dline) for k in PARSERS[ftuple].columns if CPT_format.findall(k)]))
        for pr in prlist:
            temp = vobj.prs.add()
            temp.pcode = pr
            temp.pday = 0
            temp.ctype = enums.CPT


def indexer(fname,meta,meta_list):
    index_patient = {}
    index_visit = {}
    if fname.endswith('.gz'):
        fh = gzip.open(fname)
    else:
        fh = open(fname)
    current = 0
    for line in iter(fh.readline, ''):
        if '\t' in line:
            entries = line.strip('\n').split('\t')
            index_patient[entries[0]] = current
            for entry in entries[1:]:
                vlink,state,year,ftype,dtyep,days,dline = entry.split('_')
                ftuple = (state,ftype,int(year),dtyep)
                if ftuple not in meta:
                    meta_list.append(ftuple)
                    meta[ftuple] = meta_list.index(ftuple)
                if not ftuple in PARSERS:
                    PARSERS[ftuple] = f.get_format(state,ftype,int(year),dtyep)
                index_visit[PARSERS[ftuple].get_element('KEY',dline)] = (current,meta[ftuple])
        else:
            entry = line.strip('\n')
            vlink,state,year,ftype,dtyep,days,dline = entry.split('_')
            ftuple = (state,ftype,int(year),dtyep)
            if ftuple not in meta:
                meta_list.append(ftuple)
                meta[ftuple] = meta_list.index(ftuple)
            if not ftuple in PARSERS:
                PARSERS[ftuple] = f.get_format(state,ftype,int(year),dtyep)
            index_visit[PARSERS[ftuple].get_element('KEY',dline)] = (current,meta[ftuple])
        current = fh.tell()
    return index_patient,index_visit,meta,meta_list


def fuzz_entry(entry,vobj):
    """
    Implement an entry fuzzer for generating test data
    :param entry:
    :param vobj:
    :return:
    """
    pass
from django.shortcuts import render,get_object_or_404
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required,user_passes_test
from django.core.exceptions import MultipleObjectsReturned
from django.contrib.auth.mixins import UserPassesTestMixin
from django.shortcuts import redirect
from django.views.generic import ListView,DetailView
from django.utils.decorators import method_decorator
from django.db.models import Q
import boto3,requests,json,humanize,base64,urllib
from collections import defaultdict
from chlib.entity import aggregate
from chlib import codes
from chlib.entity import enums
from chlib.entity.visit import Patient
from chlib.entity.pml_pb2 import PDXCLASSIFIER
import google
import tasks
from django.shortcuts import render
import json
# Create your views here.
try:
    JCONFIG = json.loads(file('../config.json').read())
    ML_PATH = JCONFIG["DATASETS"]["HCUPNRD"]['ROOT']+"ML/"
except:
    JCONFIG = ""
    ML_PATH = ""

def app(request):
    context = {}
    context['payload'] = {}
    payload = {
        'coder':codes.Coder(),
        'enums':[(k,vstr,vval) for k,v in enums.__dict__.iteritems() if type(v) is google.protobuf.internal.enum_type_wrapper.EnumTypeWrapper for vstr,vval in v.items()],
        'config':file('config.json').read()
    }
    context['payload'] = payload
    return render(request, 'app.html',context,using='jtlte')

def ml(request):
    context = {}
    context['payload'] = {}
    payload = {
        'dx_list':json.loads(file('{}/list.json'.format(ML_PATH)).read()),
        'selected_list':json.loads(file('{}/selected_list.json'.format(ML_PATH)).read())
    }
    context['payload'] = payload
    return render(request, 'ml.html',context,using='jtlte')


def codelist(request):
    coder = codes.Coder()
    return JsonResponse([(coder.prefix_inverse[coder.CODE_PREFIX[k]],k,v[-1]) for k,v in coder.CODES_ALL.iteritems()],safe=False)


def get_patient(db,patient_id):
    """
    get all data associated with the patient
    :param patient_id:
    :return:
    """
    result = tasks.data_get_patient.apply_async(args=[db,patient_id],queue=tasks.Q_DATA)
    result.wait()
    return result.get()


def get_databases():
    """
    get a list of databases
    :return:
    """
    examples = tasks.data_examples.apply_async(queue=tasks.Q_DATA)
    examples.wait()
    return examples.get()


@login_required
def patient_viewer(request):
    payload = {'DB_LIST':get_databases()}
    if request.GET.get('patient_id') and request.GET.get('db'):
        patient_id = request.GET.get('patient_id')
        db = request.GET.get('db')
        patient_coded = get_patient(db,patient_id)
        temp = Patient()
        v = base64.decodestring(patient_coded["data"])
        temp.ParseFromString(v)
        raw_string = temp.raw
        temp.raw = ""
        payload['coder'] = codes.Coder()
        payload['patient_obj'] = temp
        payload['next_key'] = patient_coded["next"]
        payload['patient'] = {'patient_id':patient_id,'db':db,'data':str(temp),'raw_string':raw_string}
    context = {'payload':payload}
    return render(request,'patient_viewer.html', context=context, using='jtlte')


def ml_stats_viewer(request,code):
    stats = PDXCLASSIFIER()
    stats.ParseFromString(file(ML_PATH+"/stats/{}.stats".format(code)).read())
    context = {'payload':
        {
        'stats': stats,
        'humanize': humanize,
        'Coder': codes.Coder(),
        'path': "",
        # 'los_plot_data': json.dumps(index.los_plot()),
        # 'age_plot_data': json.dumps(index.age_plot())
        }
    }
    return render(request,'ml_stats_viewer.html', context=context, using='jtlte')

def aggregate_visits_viewer(request):
    path = urllib.unquote(base64.decodestring(request.GET.get('q')))
    entry = aggregate.Aggregate()
    entry.ParseFromString(file(path).read())
    context = {'payload':
        {
        'entry': entry.obj,
        'humanize': humanize,
        'Coder': codes.Coder(),
        'path': path,
        'los_plot_data': json.dumps(entry.los_plot()),
        'age_plot_data': json.dumps(entry.age_plot())
        }
    }
    return render(request,'aggregate_visits_viewer.html', context=context, using='jtlte')


def aggregate_patients_viewer(request):
    path = urllib.unquote(base64.decodestring(request.GET.get('q')))
    entry = aggregate.PatientAggregate()
    entry.ParseFromString(file(path).read())
    context = {'payload':
        {
        'entry': entry.obj,
        'humanize': humanize,
        'Coder': codes.Coder(),
        'path': path,
        'age_plot_data': json.dumps(entry.age_plot())
        }
    }
    return render(request,'aggregate_patients_viewer.html', context=context, using='jtlte')

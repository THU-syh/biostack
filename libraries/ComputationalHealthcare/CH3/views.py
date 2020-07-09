from django.shortcuts import render,get_object_or_404
from django.http import JsonResponse
from .models import Dataset,Code,N1Group,N1Entry,N2Group,N2Entry,N4Group
from .models import N4Entry,N3Group,N3Entry,SYTCount,STCount,SCount,CodeCount,TextSearch
from django.contrib.auth.decorators import login_required,user_passes_test
from django.core.exceptions import MultipleObjectsReturned
from django.contrib.auth.mixins import UserPassesTestMixin
from django.shortcuts import redirect
from django.views.generic import ListView,DetailView
from django.utils.decorators import method_decorator
from django.db.models import Q
import boto3,requests,json,humanize,base64
from collections import defaultdict
from chlib import aggregate_visits as N1
from chlib.aggregate_edges import readmits as N2
from chlib.aggregate_edges import revisits as N3
from chlib import aggregate_patients as N4
from chlib.entity import presentation
from chlib.entity.enums import STRINGS
from chlib import codes
from chlib.entity import enums
from django.template.defaulttags import register
from chlib.entity.visit import Patient
import google
import os,logging
import boto3,botocore
from boto3.session import Session


BUCKET = 'aub3comphealth'
HEROKU = 'DATABASE_URL' in os.environ
if HEROKU:
    if 'S3_KEY' in os.environ and 'S3_SECRET' in os.environ:
        session = Session(aws_access_key_id=os.getenv("S3_KEY"),
                      aws_secret_access_key=os.getenv("S3_SECRET"),
                      region_name='us-east-1')
    else:
        session = Session()
    s3 = session.resource('s3')
    s3_client = session.client('s3')


@register.filter
def get_item(dictionary, key):
    return dictionary[key]


def get_file(dataset,fpath):
    """
    :param base_path:
    :param fpath:
    :return:
    """
    if HEROKU:
        logging.info("Retrieving {}".format(dataset.identifier+fpath))
        try:
            data = s3_client.get_object(Bucket=BUCKET, Key=dataset.identifier+fpath)['Body'].read()
        except:
            logging.exception((BUCKET,dataset.identifier+fpath))
            data = None
        return data
    else:
        return file(dataset.base_dir+fpath).read()


class LoginRequiredMixin(object):
    @method_decorator(login_required)
    def dispatch(self, *args, **kwargs):
        return super(LoginRequiredMixin, self).dispatch(*args, **kwargs)


class DatasetList(ListView):
    model = Dataset


    def get_context_data(self, **kwargs):
        context = super(DatasetList, self).get_context_data(**kwargs)
        context['datasets'] = []
        for d in Dataset.objects.all():
            temp = {}
            temp['syt'] = SYTCount.objects.filter(dataset=d)
            temp['st'] = STCount.objects.filter(dataset=d)
            temp['s'] = SCount.objects.filter(dataset=d)
            temp['entry'] = d
            context['datasets'].append(temp)
        context['Coder'] = codes.Coder()
        return context





class N4GroupByDatasetList(ListView):
    model = N4Group
    paginate_by = 50

    def get_queryset(self):
        self.dataset = get_object_or_404(Dataset, identifier=self.args[0])
        return N4Group.objects.filter(dataset=self.dataset)

    def get_context_data(self, **kwargs):
        context = super(N4GroupByDatasetList, self).get_context_data(**kwargs)
        context['dataset'] = self.dataset
        context['Coder'] = N4.N4Coder()
        return context


class CodesByDatasetList(ListView):
    model = Code
    paginate_by = 50
    template_name = 'CH3/code_dataset_list.html'

    def get_queryset(self):
        self.dataset = get_object_or_404(Dataset, identifier=self.args[0])
        return Code.objects.filter(dataset=self.dataset)

    def get_context_data(self, **kwargs):
        context = super(CodesByDatasetList, self).get_context_data(**kwargs)
        context['dataset'] = self.dataset
        return context


class CodesList(ListView):
    model = Code
    paginate_by = 50
    template_name = 'CH3/code_list.html'

    def get_context_data(self, **kwargs):
        context = super(CodesList, self).get_context_data(**kwargs)
        return context


class DatasetDetail(DetailView):
    model = Dataset

    def get_context_data(self, **kwargs):
        context = super(DatasetDetail, self).get_context_data(**kwargs)
        context['code_list'] = Code.objects.all().filter(dataset=self.object)
        return context


def marketing(request):
    return render(request, 'CH3/landing.html', {})


def library(request):
    return render(request, 'CH3/library.html', {})


def neuromine(request):
    return render(request, 'CH3/neuromine.html', {})


def index(request):
    context = {}
    context['payload'] = {}
    coder = codes.Coder()
    n3coder = N3.N3Coder()
    if request.method == 'POST' and request.POST['top-search'].strip():
        q = request.POST['top-search']
        context['payload']['q'] = q
        context['payload']['humanize'] = humanize
        context['payload']['coder'] = coder
        context['payload']['N3coder'] = n3coder
        context['payload']['results'] = TextSearch.objects.filter(description__search=q)
        context['payload']['entries_N1'] = []
        context['payload']['entries_N2'] = []
        context['payload']['entries_N3'] = []
        context['payload']['entries_N4'] = []
        context['payload']['entries_available'] = False
        for i,k in enumerate(TextSearch.objects.filter(description__search=q)):
            context['payload']['entries_available'] = True
            if i < 5:
                context['payload']['entries_N1'] += list(N1Group.objects.filter(code=str(k.code)))
                context['payload']['entries_N2'] += list(N2Group.objects.filter(code=str(k.code)))
                if k.code.startswith('P'):
                    context['payload']['entries_N3'] += list(N3Group.objects.filter(code=k.code))
                if k.code.startswith('C'):
                    context['payload']['entries_N3'] += list(N3Group.objects.filter(
                        Q(code='N3C_AS_' + k.code) | Q(code='N3C_ED_' + k.code)))
                if k.code.startswith('D'):
                    context['payload']['entries_N3'] += list(N3Group.objects.filter(
                        Q(code='N3DX_AS_' + k.code) | Q(code='N3DX_ED_' + k.code) | Q(code='N3DX_IP_' + k.code)))
                context['payload']['entries_N4'] += list(N4Group.objects.filter(code=k.code))
            else:
                break
        return render(request, 'search.html', context,using='jt')
    else:
        return redirect('/datasets/')


def index_code(request,code):
    context = {}
    context['payload'] = {}
    coder = codes.Coder()
    n3coder = N3.N3Coder()
    q = coder[code]
    context['payload']['q'] = q
    context['payload']['humanize'] = humanize
    context['payload']['coder'] = coder
    context['payload']['N3coder'] = n3coder
    context['payload']['entries_available'] = True
    context['payload']['results'] = TextSearch.objects.filter(description__search=q)
    context['payload']['entries_N1'] =list(N1Group.objects.filter(code=str(code)))
    context['payload']['entries_N2'] = N2Group.objects.filter(code=code)
    if code.startswith('P'):
        context['payload']['entries_N3'] = N3Group.objects.filter(code=code)
    if code.startswith('C'):
        context['payload']['entries_N3'] = N3Group.objects.filter(Q(code='N3C_AS_'+code) | Q(code='N3C_ED_'+code))
    if code.startswith('D'):
        context['payload']['entries_N3'] = N3Group.objects.filter(Q(code='N3DX_AS_'+code) | Q(code='N3DX_ED_'+code) | Q(code='N3DX_IP_'+code))
    context['payload']['entries_N4'] = N4Group.objects.filter(code=code)
    return render(request, 'search.html', context,using='jt')




def N1_list(request,dataset_id):
    context = {}
    context['payload'] = {}
    context['payload']['dataset'] = get_object_or_404(Dataset, identifier=dataset_id)
    context['payload']['Coder'] = N1.N1Coder()
    context['payload']['entries'] = list(N1Group.objects.filter(dataset=context['payload']['dataset'])
                                         .filter(Q(count__gt=1) | Q(pediatric_count__gt=1)))
    return render(request, 'N1/list.html', context=context, using='jt')


def N1_group_detail(request,dataset_id,code):
    dataset = Dataset.objects.get(identifier=dataset_id)
    split = N1Group.objects.get(code=code,dataset=dataset)
    exclusions = json.loads(split.excluded)
    if code.startswith('P'):
        groups = N1Entry.objects.all().filter(Q(initial=code) | Q(sub=code)).filter(dataset=dataset)
        pr_type = True
    else:
        groups = N1Entry.objects.all().filter(dx=code).filter(dataset=dataset)
        pr_type = False
    context = {
        'payload':
            {'Coder': N1.N1Coder(),
             'pr_type':pr_type,
             'code':code,
             'dataset':dataset,
             'groups':groups,
             'split':split,
             'exclusions':exclusions}
        }
    return render(request, 'N1/group.html', context=context, using='jt')


def get_N1_payload(dataset_id,key):
    dataset = Dataset.objects.get(identifier=dataset_id)
    entry_row = N1Entry.objects.get(key=key,dataset=dataset)
    entry = N1.Entry()
    entry.ParseFromString(get_file(dataset,'/RESULT/'+entry_row.filename))
    return {'Coder': N1.N1Coder(),
             'entry':entry.obj,
             'key':key,
             'dataset':dataset,
             'humanize':humanize,
             'delta_plot_data': json.dumps(entry.delta_subset_plot()),
             'los_plot_data': json.dumps(entry.los_plot()),
             'age_plot_data': json.dumps(entry.age_plot()),
             'delta_multi_plot_data': json.dumps(entry.plot_data()),
             'subset_table': presentation.subset_table,
             'dropdown': presentation.get_dropdown(entry.delta_subset_plot()),
             'subset_entry_table': presentation.subset_entry_table,
             }


def N1_entry(request,dataset_id,key):
    context = {'payload':get_N1_payload(dataset_id,key)}
    return render(request, 'N1/entry.html', context=context, using='jt')


def N1_compare(request,dataset_id_1,key_1,dataset_id_2,key_2):
    payload = {'left': get_N1_payload(dataset_id_1,key_1),
                'right': get_N1_payload(dataset_id_2,key_2),
                'Coder': N1.N1Coder(),
                'combine_lr':presentation.combine_lr,
                'combiner': presentation.combine_tables,
                'combiner_dx': presentation.combine_dx
                }
    context = {'payload':payload}
    return render(request, 'N1/compare.html', context=context, using='jt')


def N2_list(request,dataset_id):
    context = {}
    context['payload'] = {}
    context['payload']['dataset'] = get_object_or_404(Dataset, identifier=dataset_id)
    context['payload']['Coder'] = N2.N2Coder()
    context['payload']['entries'] = list(N2Group.objects.filter(dataset=context['payload']['dataset']).filter(Q(index_count__gt=1)))
    return render(request, 'N2/list.html', context=context, using='jt')


def N2_group_detail(request,dataset_id,code):
    dataset = Dataset.objects.get(identifier=dataset_id)
    split = N2Group.objects.get(code=code,dataset=dataset)
    index_exclusions = json.loads(split.excluded_index)
    entry = N2.Node()
    entry.ParseFromString(get_file(dataset,'/RESULT/' + split.index_filename))
    edge_exclusions = json.loads(split.excluded_edges)
    ulinked_exclusions = json.loads(split.excluded_unlinked)
    groups = N2Entry.objects.all().filter(split_code=code).filter(dataset=dataset)
    context = {
        'payload':
            {'Coder': N2.N2Coder(),
             'code':code,
             'split_code':code,
             'node':entry.obj,
             'dataset':dataset,
             'edges':groups,
             'split':split,
             'exclusions_index':index_exclusions,
             'exclusions_unlinked':ulinked_exclusions,
             'exclusions_edge':edge_exclusions,
             }
        }
    return render(request, 'N2/group.html', context=context, using='jt')


def N2_entry(request,dataset_id,key):
    dataset = Dataset.objects.get(identifier=dataset_id)
    entry_row = N2Entry.objects.get(key=key,dataset=dataset)
    entry_type = entry_row.entry_type
    fname = dataset.base_dir +'/RESULT/' + entry_row.filename
    if entry_type == 'edge':
        edge = N2.Edge()
        node = N2.Node()
        node_row = N2Entry.objects.get(key=entry_row.initial,dataset=dataset,entry_type='node',linked=True)
        node_fname = dataset.base_dir +'/RESULT/' + node_row.filename
        edge.ParseFromString(get_file(dataset,'/RESULT/' + entry_row.filename))
        node.ParseFromString(get_file(dataset,'/RESULT/' + node_row.filename))
        context = {
            'payload':{
            'Coder': N2.N2Coder(),
            'edge': edge.obj,
            'combiner': presentation.combine_tables,
            'combiner_dx': presentation.combine_dx,
            'node': node.obj,
            'delta_plot_data': json.dumps(edge.plot_data()),
            'delta_week_plot_data': json.dumps(edge.week_plot_data()),
            'los_plot_data_initial': json.dumps(edge.los_plot_initial()),
            'los_plot_data_sub': json.dumps(edge.los_plot_sub()),
            'los_plot_data_discharged': json.dumps(node.los_plot_discharged()),
            'age_plot_data_initial': json.dumps(edge.age_plot_initial()),
            'age_plot_data_sub': json.dumps(edge.age_plot_sub()),
            'age_plot_data_discharged': json.dumps(node.age_plot_discharged()),
            'dataset': dataset,
            'humanize': humanize
            }
        }
        return render(request, 'N2/edge.html', context=context, using='jt')
    else:
        payload = {}
        return 404


def N3_list(request,dataset_id):
    context = {}
    context['payload'] = {}
    context['payload']['dataset'] = get_object_or_404(Dataset, identifier=dataset_id)
    context['payload']['Coder'] = N3.N3Coder()
    context['payload']['entries'] = list(N3Group.objects.filter(dataset=context['payload']['dataset']).filter(Q(index_count__gt=1)))
    return render(request, 'N3/list.html', context=context, using='jt')


def N3_group_detail(request,dataset_id,code):
    dataset = Dataset.objects.get(identifier=dataset_id)
    split = N3Group.objects.get(code=code,dataset=dataset)
    index_exclusions = json.loads(split.excluded_index)
    entry = N3.Node()
    entry.ParseFromString(get_file(dataset,'/RESULT/' + split.index_filename))
    edge_exclusions = json.loads(split.excluded_edges)
    ulinked_exclusions = json.loads(split.excluded_unlinked)
    groups = N3Entry.objects.all().filter(split_code=code).filter(dataset=dataset)
    context = {
        'payload':
            {'Coder': N3.N3Coder(),
             'code':code,
             'split_code':code,
             'node':entry.obj,
             'dataset':dataset,
             'edges':groups,
             'split':split,
             'exclusions_index':index_exclusions,
             'exclusions_unlinked':ulinked_exclusions,
             'exclusions_edge':edge_exclusions,
             }
        }
    return render(request, 'N3/group.html', context=context, using='jt')


def N3_entry(request,dataset_id,key):
    dataset = Dataset.objects.get(identifier=dataset_id)
    entry_row = N3Entry.objects.get(key=key,dataset=dataset)
    entry_type = entry_row.entry_type
    fname = dataset.base_dir +'/RESULT/' + entry_row.filename
    if entry_type == 'edge':
        edge = N3.Edge()
        node = N3.Node()
        node_row = N3Entry.objects.get(key=entry_row.initial,dataset=dataset,entry_type='node',linked=True)
        node_fname = dataset.base_dir +'/RESULT/' + node_row.filename
        edge.ParseFromString(get_file(dataset,'/RESULT/' + entry_row.filename))
        node.ParseFromString(get_file(dataset,'/RESULT/' + node_row.filename))
        context = {
            'payload':{
            'Coder': N3.N3Coder(),
            'edge': edge.obj,
            'combiner': presentation.combine_tables,
            'combiner_dx': presentation.combine_dx,
            'node': node.obj,
            'delta_plot_data': json.dumps(edge.plot_data()),
            'delta_week_plot_data': json.dumps(edge.week_plot_data()),
            'los_plot_data_initial': json.dumps(edge.los_plot_initial()),
            'los_plot_data_sub': json.dumps(edge.los_plot_sub()),
            'los_plot_data_discharged': json.dumps(node.los_plot_discharged()),
            'age_plot_data_initial': json.dumps(edge.age_plot_initial()),
            'age_plot_data_sub': json.dumps(edge.age_plot_sub()),
            'age_plot_data_discharged': json.dumps(node.age_plot_discharged()),
            'dataset': dataset,
            'humanize': humanize
            }
        }
        return render(request, 'N3/edge.html', context=context, using='jt')
    else:
        payload = {}
        return 404


@login_required
def N4_group_detail(request,dataset_id,code):
    dataset = Dataset.objects.get(identifier=dataset_id)
    node = N4Group.objects.get(code=code,dataset=dataset)
    entry = N4.Edge()
    entry.ParseFromString(get_file(dataset,'/RESULT/' + node.filename))
    groups = N4Entry.objects.all().filter(index=code).filter(dataset=dataset)
    context = {
        'payload':
            {'Coder': N4.N4Coder(),
             'code': code,
             'split_code': code,
             'node_entry':entry.obj,
             'dataset':dataset,
             'edges':groups,
             }
        }
    return render(request, 'N4/group.html', context=context, using='jt')


@login_required
def N4_entry(request,dataset_id,key):
    dataset = Dataset.objects.get(identifier=dataset_id)
    entry_row = N4Entry.objects.get(key=key,dataset=dataset)
    fname = dataset.base_dir +'/RESULT/' + entry_row.filename
    edge = N4.Edge()
    edge.ParseFromString(get_file(dataset,'/RESULT/' + entry_row.filename))
    context = {
        'payload':{
        'Coder': N4.N4Coder(),
        'edge_row': entry_row,
        'edge': edge.obj,
        'combiner': presentation.combine_tables,
        'combiner_dx': presentation.combine_dx,
        'dataset': dataset,
        'humanize': humanize,
        'age_plot_data': json.dumps(edge.age_plot()),
        'excluded':json.loads(entry_row.excluded)
        }
    }
    return render(request, 'N4/edge.html', context=context, using='jt')


def config(request):
    context = {}
    payload = {
        'coder':codes.Coder(),
        'enums':[(k,vstr,vval) for k,v in enums.__dict__.iteritems() if type(v) is google.protobuf.internal.enum_type_wrapper.EnumTypeWrapper for vstr,vval in v.items()],
    }
    context['payload'] = payload
    return render(request, 'config.html', context=context, using='jt')


def codelist(request):
    coder = codes.Coder()
    return JsonResponse([(coder.prefix_inverse[coder.CODE_PREFIX[k]],k,v[-1]) for k,v in coder.CODES_ALL.iteritems()],safe=False)


def code_detail(request,code):
    by_dataset = {}
    for d in Dataset.objects.all():
        by_dataset[d.identifier] = {'counts':[], 'dataset': d}
    for k in CodeCount.objects.filter(code=code):
        by_dataset[k.dataset_identifier]['counts'].append(k)
    for k in by_dataset:
        by_dataset[k]['visits'] = sum([c.count for c in by_dataset[k]['counts'] if c.state == 'ALL' and c.code_type != 'pdx'])
    context = {}
    payload = {
        'code':code,
        'Coder':codes.Coder(),
        'datasets' : by_dataset.values(),
        'visit_type_str': STRINGS,
        'code_type_str': { "pdx": "Primary Diagnosis", 'dx':"Diagnosis", "pr":"Procedure", "drg":"DRG"}
    }
    context['payload'] = payload
    return render(request, 'code.html', context=context, using='jt')

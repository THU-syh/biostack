from __future__ import unicode_literals
from django.db import models
from django.contrib.postgres.fields import ArrayField
from chlib.entity.enums import CTYPE


class Dataset(models.Model):
    identifier = models.CharField(max_length=100)
    name = models.CharField(max_length=100)
    linked = models.BooleanField(default=False)
    base_dir = models.CharField(max_length=200)
    states = ArrayField(models.CharField(max_length=10))
    years = ArrayField(models.IntegerField())
    patients_count = models.IntegerField(default=0)
    linked_count = models.IntegerField(default=0)
    unlinked_count = models.IntegerField(default=0)
    aggregate_visits = models.BooleanField(default=False)
    aggregate_readmits = models.BooleanField(default=False)
    aggregate_revisits = models.BooleanField(default=False)
    aggregate_patients = models.BooleanField(default=False)


class SCount(models.Model):
    dataset = models.ForeignKey(Dataset)
    state = models.CharField(max_length=10)
    patients_count = models.IntegerField(default=0)
    linked_count = models.IntegerField(default=0)
    unlinked_count = models.IntegerField(default=0)


class STCount(models.Model):
    dataset = models.ForeignKey(Dataset)
    state = models.CharField(max_length=10)
    visit_type = models.PositiveSmallIntegerField()
    linked = models.BooleanField(default=False)
    count = models.IntegerField(default=0)


class SYTCount(models.Model):
    dataset = models.ForeignKey(Dataset)
    state = models.CharField(max_length=10)
    year = models.IntegerField()
    linked = models.BooleanField(default=False)
    visit_type = models.PositiveSmallIntegerField()
    count = models.IntegerField(default=0)


class Code(models.Model):
    code = models.CharField(max_length=100)
    description = models.TextField(max_length=300)
    indexed = models.BooleanField(default=False)
    dataset = models.ForeignKey(Dataset)
    code_type = models.CharField(max_length=5,default="")


class CodeCount(models.Model):
    dataset_identifier = models.CharField(max_length=10)
    code = models.CharField(max_length=50)
    code_type = models.CharField(max_length=5)
    count = models.IntegerField(default=0)
    visit_type = models.PositiveSmallIntegerField()
    state = models.CharField(max_length=10)
    year = models.IntegerField()
    linked = models.BooleanField(default=False)


class N1Group(models.Model):
    dataset = models.ForeignKey(Dataset)
    code = models.CharField(max_length=50)
    count = models.IntegerField(default=0)
    delta_median = models.IntegerField(default=-1)
    filename = models.CharField(max_length=200, default="")
    pediatric = models.BooleanField(default=False)
    pediatric_count = models.IntegerField(default=0)
    pediatric_delta_median = models.IntegerField(default=-1)
    key = models.CharField(max_length=100)
    pediatric_key = models.CharField(max_length=100)
    excluded = models.TextField()


class N2Group(models.Model):
    dataset = models.ForeignKey(Dataset)
    code = models.CharField(max_length=50)
    index_count = models.IntegerField(default=0)
    unlinked_count = models.IntegerField(default=0)
    readmit_30_count = models.IntegerField(default=0)
    readmit_90_count = models.IntegerField(default=0)
    index_key = models.CharField(max_length=100)
    computed = models.BooleanField(default=False)
    excluded_index = models.TextField()
    excluded_edges = models.TextField()
    excluded_unlinked = models.TextField()
    index_filename = models.CharField(max_length=100)


class N3Group(models.Model):
    dataset = models.ForeignKey(Dataset)
    code = models.CharField(max_length=50)
    index_count = models.IntegerField(default=0)
    unlinked_count = models.IntegerField(default=0)
    revisit_30_count = models.IntegerField(default=0)
    revisit_90_count = models.IntegerField(default=0)
    index_key = models.CharField(max_length=100)
    computed = models.BooleanField(default=False)
    excluded_index = models.TextField()
    excluded_edges = models.TextField()
    excluded_unlinked = models.TextField()
    index_filename = models.CharField(max_length=100)


class N1Entry(models.Model):
    dataset = models.ForeignKey(Dataset)
    initial = models.CharField(max_length=50)
    sub = models.CharField(max_length=50)
    dx = models.CharField(max_length=50)
    count = models.IntegerField(default=0)
    delta_median = models.IntegerField(default=-1)
    filename = models.CharField(max_length=200,default="")
    pediatric = models.BooleanField(default=False)
    key = models.CharField(max_length=100)


class N2Entry(models.Model):
    dataset = models.ForeignKey(Dataset)
    split_code = models.CharField(max_length=50)
    initial = models.CharField(max_length=50)
    sub = models.CharField(max_length=50)
    count = models.IntegerField(default=0)
    key = models.CharField(max_length=100)
    filename = models.CharField(max_length=200,default="")
    entry_type = models.CharField(max_length=10)
    linked = models.BooleanField(default=True)


class N3Entry(models.Model):
    dataset = models.ForeignKey(Dataset)
    split_code = models.CharField(max_length=50)
    initial = models.CharField(max_length=50)
    sub = models.CharField(max_length=50)
    count = models.IntegerField(default=0)
    key = models.CharField(max_length=100)
    filename = models.CharField(max_length=200,default="")
    entry_type = models.CharField(max_length=10)
    linked = models.BooleanField(default=True)

class N4Group(models.Model):
    dataset = models.ForeignKey(Dataset)
    code = models.CharField(max_length=50)
    patients = models.IntegerField(default=0)
    unlinked_count = models.IntegerField(default=0)
    key = models.CharField(max_length=100)
    filename = models.CharField(max_length=200,default="")


class N4Entry(models.Model):
    dataset = models.ForeignKey(Dataset)
    split_code = models.CharField(max_length=50)
    index = models.CharField(max_length=50)
    sub = models.CharField(max_length=50)
    patients = models.IntegerField(default=0)
    key = models.CharField(max_length=100)
    filename = models.CharField(max_length=200,default="")
    excluded = models.TextField()


class TextSearch(models.Model):
    code = models.CharField(max_length=30)
    description = models.TextField()
    datasets_count = models.IntegerField()
    code_type = models.CharField(max_length=5,default="")
from django.contrib import admin
from .models import Dataset,CodeCount,Code,SCount,STCount,\
    SYTCount,N1Entry,N1Group,N2Entry,N2Group,N4Group,N4Entry,\
    N3Entry,N3Group


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    pass


@admin.register(CodeCount)
class CodeCountAdmin(admin.ModelAdmin):
    pass


@admin.register(Code)
class CodeAdmin(admin.ModelAdmin):
    pass


@admin.register(SCount)
class SCountAdmin(admin.ModelAdmin):
    pass


@admin.register(STCount)
class STCountAdmin(admin.ModelAdmin):
    pass


@admin.register(SYTCount)
class SYTCountAdmin(admin.ModelAdmin):
    pass


@admin.register(N1Entry)
class N1EntryAdmin(admin.ModelAdmin):
    pass


@admin.register(N1Group)
class N1GroupAdmin(admin.ModelAdmin):
    pass


@admin.register(N2Entry)
class N2EntryAdmin(admin.ModelAdmin):
    pass


@admin.register(N2Group)
class N2GroupAdmin(admin.ModelAdmin):
    pass


@admin.register(N3Entry)
class N3EntryAdmin(admin.ModelAdmin):
    pass


@admin.register(N3Group)
class N3GroupAdmin(admin.ModelAdmin):
    pass


@admin.register(N4Entry)
class N4EntryAdmin(admin.ModelAdmin):
    pass


@admin.register(N4Group)
class N4GroupAdmin(admin.ModelAdmin):
    pass



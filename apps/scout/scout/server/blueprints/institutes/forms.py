# -*- coding: utf-8 -*-
from flask_wtf import FlaskForm
from wtforms.widgets import TextInput
from wtforms import (
    IntegerField,
    SelectMultipleField,
    SubmitField,
    DecimalField,
    TextField,
    validators,
    Field,
)
from scout.constants import PHENOTYPE_GROUPS


class NonValidatingSelectMultipleField(SelectMultipleField):
    """Necessary to skip validation of dynamic multiple selects in form"""

    def pre_validate(self, form):
        pass


class InstituteForm(FlaskForm):
    """ Instutute-specif settings """

    hpo_tuples = []
    for key in PHENOTYPE_GROUPS.keys():
        option_name = " ".join(
            [key, ",", PHENOTYPE_GROUPS[key]["name"], "(", PHENOTYPE_GROUPS[key]["abbr"], ")",]
        )
        hpo_tuples.append((option_name, option_name))

    display_name = TextField(
        "Institute display name",
        validators=[validators.InputRequired(), validators.Length(min=2, max=100)],
    )
    sanger_emails = NonValidatingSelectMultipleField(
        "Sanger recipients", validators=[validators.Optional()]
    )
    coverage_cutoff = IntegerField(
        "Coverage cutoff", validators=[validators.Optional(), validators.NumberRange(min=1)],
    )
    frequency_cutoff = DecimalField(
        "Frequency cutoff",
        validators=[
            validators.Optional(),
            validators.NumberRange(min=0, message="Number must be positive"),
        ],
    )

    pheno_group = TextField("New phenotype group", validators=[validators.Optional()])
    pheno_abbrev = TextField("Abbreviation", validators=[validators.Optional()])

    pheno_groups = NonValidatingSelectMultipleField("Custom phenotype groups", choices=hpo_tuples)
    cohorts = NonValidatingSelectMultipleField(
        "Available patient cohorts", validators=[validators.Optional()]
    )
    institutes = NonValidatingSelectMultipleField("Institutes to share cases with", choices=[])
    submit_btn = SubmitField("Save settings")


# make a base class or other utility with this instead..
class TagListField(Field):
    widget = TextInput()

    def _value(self):
        if self.data:
            return ", ".join(self.data)

        return ""

    def process_formdata(self, valuelist):
        if valuelist:
            self.data = [x.strip() for x in valuelist[0].split(",") if x.strip()]
        else:
            self.data = []


class GeneVariantFiltersForm(FlaskForm):
    """Base FiltersForm for SNVs"""

    variant_type = SelectMultipleField(choices=[("clinical", "clinical"), ("research", "research")])
    hgnc_symbols = TagListField("HGNC Symbols/Ids (case sensitive)")
    filter_variants = SubmitField(label="Filter variants")
    rank_score = IntegerField(default=15)
    phenotype_terms = TagListField("HPO terms")
    phenotype_groups = TagListField("Phenotype groups")
    similar_case = TagListField("Phenotypically similar case")
    cohorts = TagListField("Cohorts")

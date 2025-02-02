# -*- coding: utf-8 -*-
import logging
import pymongo

from flask import (
    Blueprint,
    render_template,
    flash,
    redirect,
    request,
    Response,
    url_for,
)
from flask_login import current_user
from werkzeug.datastructures import Headers

from . import controllers
from scout.constants import (
    PHENOTYPE_GROUPS,
    CASEDATA_HEADER,
    CLINVAR_HEADER,
    ACMG_COMPLETE_MAP,
    ACMG_MAP,
)
from scout.server.extensions import store
from scout.server.utils import user_institutes, templated, institute_and_case
from .forms import InstituteForm, GeneVariantFiltersForm

LOG = logging.getLogger(__name__)

blueprint = Blueprint(
    "overview",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/overview/static",
)


@blueprint.route("/overview")
def institutes():
    """Display a list of all user institutes."""
    institute_objs = user_institutes(store, current_user)
    institutes = []
    for ins_obj in institute_objs:
        sanger_recipients = []
        for user_mail in ins_obj.get("sanger_recipients", []):
            user_obj = store.user(user_mail)
            if not user_obj:
                continue
            sanger_recipients.append(user_obj["name"])
        institutes.append(
            {
                "display_name": ins_obj["display_name"],
                "internal_id": ins_obj["_id"],
                "coverage_cutoff": ins_obj.get("coverage_cutoff", "None"),
                "sanger_recipients": sanger_recipients,
                "frequency_cutoff": ins_obj.get("frequency_cutoff", "None"),
                "phenotype_groups": ins_obj.get("phenotype_groups", PHENOTYPE_GROUPS),
                "case_count": sum(1 for i in store.cases(collaborator=ins_obj["_id"])),
            }
        )

    data = dict(institutes=institutes)
    return render_template("overview/institutes.html", **data)


@blueprint.route("/<institute_id>/cases")
@templated("overview/cases.html")
def cases(institute_id):
    """Display a list of cases for an institute."""

    institute_obj = institute_and_case(store, institute_id)
    query = request.args.get("query")

    limit = 100
    if request.args.get("limit"):
        limit = int(request.args.get("limit"))

    skip_assigned = request.args.get("skip_assigned")
    is_research = request.args.get("is_research")
    all_cases = store.cases(
        collaborator=institute_id,
        name_query=query,
        skip_assigned=skip_assigned,
        is_research=is_research,
    )

    sort_by = request.args.get("sort")
    sort_order = request.args.get("order") or "asc"
    if sort_by:
        pymongo_sort = pymongo.ASCENDING
        if sort_order == "desc":
            pymongo_sort = pymongo.DESCENDING
        if sort_by == "analysis_date":
            all_cases.sort("analysis_date", pymongo_sort)
        elif sort_by == "track":
            all_cases.sort("track", pymongo_sort)
        elif sort_by == "status":
            all_cases.sort("status", pymongo_sort)

    LOG.debug("Prepare all cases")

    prioritized_cases = store.prioritized_cases(institute_id=institute_id)

    data = controllers.cases(store, all_cases, prioritized_cases, limit)
    data["sort_order"] = sort_order
    data["sort_by"] = sort_by
    data["nr_cases"] = store.nr_cases(institute_id=institute_id)

    sanger_unevaluated = controllers.get_sanger_unevaluated(store, institute_id, current_user.email)
    if len(sanger_unevaluated) > 0:
        data["sanger_unevaluated"] = sanger_unevaluated

    return dict(
        institute=institute_obj,
        skip_assigned=skip_assigned,
        is_research=is_research,
        query=query,
        **data,
    )


@blueprint.route("/<institute_id>/causatives")
@templated("overview/causatives.html")
def causatives(institute_id):
    institute_obj = institute_and_case(store, institute_id)
    query = request.args.get("query", "")
    hgnc_id = None
    if "|" in query:
        # filter accepts an array of IDs. Provide an array with one ID element
        try:
            hgnc_id = [int(query.split(" | ", 1)[0])]
        except ValueError:
            flash("Provided gene info could not be parsed!", "warning")

    variants = store.check_causatives(institute_obj=institute_obj, limit_genes=hgnc_id)
    if variants:
        variants.sort("hgnc_symbols", pymongo.ASCENDING)
    all_variants = {}
    all_cases = {}
    for variant_obj in variants:
        if variant_obj["case_id"] not in all_cases:
            case_obj = store.case(variant_obj["case_id"])
            all_cases[variant_obj["case_id"]] = case_obj
        else:
            case_obj = all_cases[variant_obj["case_id"]]

        if variant_obj["variant_id"] not in all_variants:
            all_variants[variant_obj["variant_id"]] = []

        all_variants[variant_obj["variant_id"]].append((case_obj, variant_obj))

    acmg_map = {key: ACMG_COMPLETE_MAP[value] for key, value in ACMG_MAP.items()}

    return dict(institute=institute_obj, variant_groups=all_variants, acmg_map=acmg_map)


@blueprint.route("/<institute_id>/gene_variants", methods=["GET", "POST"])
@templated("overview/gene_variants.html")
def gene_variants(institute_id):
    """Display a list of SNV variants."""
    page = int(request.form.get("page", 1))

    institute_obj = institute_and_case(store, institute_id)

    # populate form, conditional on request method
    if request.method == "POST":
        form = GeneVariantFiltersForm(request.form)
    else:
        form = GeneVariantFiltersForm(request.args)

    if form.variant_type.data == []:
        form.variant_type.data = ["clinical"]

    variant_type = form.data.get("variant_type")

    # check if supplied gene symbols exist
    hgnc_symbols = []
    non_clinical_symbols = []
    not_found_symbols = []
    not_found_ids = []
    data = {}
    if (form.hgnc_symbols.data) and len(form.hgnc_symbols.data) > 0:
        is_clinical = form.data.get("variant_type", "clinical") == "clinical"
        # clinical_symbols = store.clinical_symbols(case_obj) if is_clinical else None
        for hgnc_symbol in form.hgnc_symbols.data:
            if hgnc_symbol.isdigit():
                hgnc_gene = store.hgnc_gene(int(hgnc_symbol))
                if hgnc_gene is None:
                    not_found_ids.append(hgnc_symbol)
                else:
                    hgnc_symbols.append(hgnc_gene["hgnc_symbol"])
            elif store.hgnc_genes(hgnc_symbol).count() == 0:
                not_found_symbols.append(hgnc_symbol)
            # elif is_clinical and (hgnc_symbol not in clinical_symbols):
            #     non_clinical_symbols.append(hgnc_symbol)
            else:
                hgnc_symbols.append(hgnc_symbol)

        if not_found_ids:
            flash("HGNC id not found: {}".format(", ".join(not_found_ids)), "warning")
        if not_found_symbols:
            flash(
                "HGNC symbol not found: {}".format(", ".join(not_found_symbols)), "warning",
            )
        if non_clinical_symbols:
            flash(
                "Gene not included in clinical list: {}".format(", ".join(non_clinical_symbols)),
                "warning",
            )
        form.hgnc_symbols.data = hgnc_symbols

        LOG.debug("query {}".format(form.data))

        variants_query = store.gene_variants(
            query=form.data, institute_id=institute_id, category="snv", variant_type=variant_type,
        )

        data = controllers.gene_variants(store, variants_query, institute_id, page)

    return dict(institute=institute_obj, form=form, page=page, **data)


@blueprint.route("/overview/<institute_id>/settings", methods=["GET", "POST"])
def institute_settings(institute_id):
    """Show institute settings page"""

    if institute_id not in current_user.institutes and current_user.is_admin is False:
        flash(
            "Current user doesn't have the permission to modify this institute", "warning",
        )
        return redirect(request.referrer)

    institute_obj = store.institute(institute_id)
    form = InstituteForm(request.form)

    # if institute is to be updated
    if request.method == "POST" and form.validate_on_submit():
        institute_obj = controllers.update_institute_settings(store, institute_obj, request.form)
        if isinstance(institute_obj, dict):
            flash("institute was updated ", "success")
        else:  # an error message was retuned
            flash(institute_obj, "warning")
            return redirect(request.referrer)

    data = controllers.institute(store, institute_id)
    default_phenotypes = controllers.populate_institute_form(form, institute_obj)

    return render_template(
        "/overview/institute_settings.html",
        form=form,
        default_phenotypes=default_phenotypes,
        panel=1,
        **data,
    )


@blueprint.route("/overview/<institute_id>/users", methods=["GET"])
def institute_users(institute_id):
    """Should institute users list"""

    if institute_id not in current_user.institutes and current_user.is_admin is False:
        flash(
            "Current user doesn't have the permission to modify this institute", "warning",
        )
        return redirect(request.referrer)
    data = controllers.institute(store, institute_id)
    return render_template("/overview/users.html", panel=2, **data)


@blueprint.route("/<submission>/<case>/rename/<old_name>", methods=["POST"])
def clinvar_rename_casedata(submission, case, old_name):
    """Rename one or more casedata individuals belonging to the same clinvar submission, same case"""

    new_name = request.form.get("new_name")
    controllers.update_clinvar_sample_names(
        store, submission, case, old_name, new_name,
    )
    return redirect(request.referrer)


@blueprint.route("/<institute_id>/<submission>/update_status", methods=["POST"])
def clinvar_update_submission(institute_id, submission):
    """Update a submission status to open/closed, register an official SUB number or delete the entire submission"""

    controllers.update_clinvar_submission_status(store, request, institute_id, submission)
    return redirect(request.referrer)


@blueprint.route("/<submission>/<object_type>", methods=["POST"])
def clinvar_delete_object(submission, object_type):
    """Delete a single object (variant_data or case_data) associated with a clinvar submission"""

    store.delete_clinvar_object(
        object_id=request.form.get("delete_object"),
        object_type=object_type,
        submission_id=submission,
    )
    return redirect(request.referrer)


@blueprint.route("/<submission>/download/<csv_type>/<clinvar_id>", methods=["GET"])
def clinvar_download_csv(submission, csv_type, clinvar_id):
    """Download a csv (Variant file or CaseData file) for a clinVar submission"""

    def generate_csv(header, lines):
        """Return downloaded header and lines with quoted fields"""
        yield header + "\n"
        for line in lines:
            yield line + "\n"

    clinvar_file_data = controllers.clinvar_submission_file(store, submission, csv_type, clinvar_id)

    if clinvar_file_data is None:
        return redirect(request.referrer)

    headers = Headers()
    headers.add(
        "Content-Disposition", "attachment", filename=clinvar_file_data[0],
    )
    return Response(
        generate_csv(",".join(clinvar_file_data[1]), clinvar_file_data[2]),
        mimetype="text/csv",
        headers=headers,
    )


@blueprint.route("/<institute_id>/clinvar_submissions", methods=["GET"])
@templated("overview/clinvar_submissions.html")
def clinvar_submissions(institute_id):
    """Handle clinVar submission objects and files"""

    institute_obj = institute_and_case(store, institute_id)

    data = {
        "submissions": controllers.clinvar_submissions(store, institute_id),
        "institute": institute_obj,
        "variant_header_fields": CLINVAR_HEADER,
        "casedata_header_fields": CASEDATA_HEADER,
    }
    return data

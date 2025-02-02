# -*- coding: utf-8 -*-
import datetime
from flask import url_for, current_app, get_template_attribute
from flask_login import current_user
from pymongo import ReturnDocument

from scout.demo import delivery_report_path
from scout.server.blueprints.cases import controllers
from scout.server.extensions import store
from scout.server.blueprints.cases.views import (
    parse_raw_gene_symbols,
    parse_raw_gene_ids,
)

TEST_TOKEN = "test_token"


def test_parse_raw_gene_symbols(app):
    """Test parse gene symbols"""

    # GIVEN a list of autocompleted gene symbols
    gene_symbols = ["MUTYH |POT1", "POT1 0.1|APC|PMS2"]

    # WHEN converting to hgnc_ids
    hgnc_symbols = parse_raw_gene_symbols(gene_symbols)

    # THEN the appropriate set of hgnc_symbols should be returned
    assert hgnc_symbols == {"APC", "MUTYH", "PMS2", "POT1"}


def test_parse_raw_gene_ids(app):
    """ Test parse gene symbols"""

    # GIVEN a list of autocompleted gene symbols
    gene_symbols = ["1234 | SYM (OLDSYM, SYM)", "4321 | MYS (OLDMYS, MYS)"]

    # WHEN converting to hgnc_ids
    hgnc_ids = parse_raw_gene_ids(gene_symbols)

    # THEN the appropriate set of hgnc_ids should be returned
    assert hgnc_ids == {1234, 4321}


def test_sidebar_macro(app, institute_obj, case_obj):
    """test the case sidebar macro"""

    # GIVEN a case with several delivery reports, both in "delivery_report" field and "analyses" field
    today = datetime.datetime.now()
    one_year_ago = today - datetime.timedelta(days=365)
    five_years_ago = today - datetime.timedelta(days=5 * 365)
    new_report = "new_delivery_report.html"
    case_analyses = [
        dict(
            # fresh analysis from today
            date=today,
            delivery_report=new_report,
        ),
        dict(
            # old analysis is 1 year old, missing the report
            date=one_year_ago,
            delivery_report=None,
        ),
        dict(
            # ancient analysis is 5 year old
            date=five_years_ago,
            delivery_report="ancient_delivery_report.html",
        ),
    ]
    # update test case with the analyses above
    updated_case = store.case_collection.find_one_and_update(
        {"_id": case_obj["_id"]},
        {
            "$set": {
                "analysis_date": today,
                "delivery_report": new_report,
                "analyses": case_analyses,
            }
        },
        return_document=ReturnDocument.AFTER,
    )

    # GIVEN an initialized app
    with app.test_client() as client:
        # WHEN the case sidebar macro is called
        macro = get_template_attribute("cases/collapsible_actionbar.html", "action_bar")
        html = macro(institute_obj, updated_case)

        # It should show the expected items:
        assert "Reports" in html
        assert "General" in html
        assert "mtDNA report" in html

        # only 2 delivery reports should be showed
        today = str(today).split(" ")[0]
        assert f"Delivery ({today})" in html

        five_years_ago = str(five_years_ago).split(" ")[0]
        assert f"Delivery ({five_years_ago})" in html

        # The analysis with missing report should not be shown
        one_year_ago = str(one_year_ago).split(" ")[0]
        assert f"Delivery ({one_year_ago})" not in html

        assert f"Genome build {case_obj['genome_build']}" in html
        assert f"Rank model" in html
        assert f"Status: {case_obj['status'].capitalize()}" in html
        assert "Assignees" in html
        assert "Research list" in html
        assert "Reruns" in html
        assert "Share case" in html


def test_update_cancer_case_sample(app, user_obj, institute_obj, cancer_case_obj):
    # GIVEN an initialized app
    # GIVEN a valid user and institute

    # And a cancer case with cancer samples data
    old_tumor_purity = cancer_case_obj["individuals"][0]["tumor_purity"]
    old_tumor_type = cancer_case_obj["individuals"][0]["tumor_type"]
    old_tissue_type = "unknown"
    assert old_tumor_purity
    assert old_tumor_type

    cancer_case_obj["individuals"][0]["tissue_type"] = old_tissue_type
    cancer_case_obj["updated_at"] = datetime.datetime.now()
    store.case_collection.insert_one(cancer_case_obj)

    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # WHEN posting a request with info for updating one of the case samples:
        ind_id = cancer_case_obj["individuals"][0]["individual_id"]

        form_data = {
            "update_ind": ind_id,
            ".".join(["tumor_type.", ind_id]): "Desmoid Tumor",
            ".".join(["tissue_type", ind_id]): "cell line",
            ".".join(["tumor_purity", ind_id]): "0.4",
        }

        resp = client.post(
            url_for(
                "cases.update_cancer_sample",
                institute_id=institute_obj["internal_id"],
                case_name=cancer_case_obj["display_name"],
            ),
            data=form_data,
        )

        # THEN the returned HTML page should redirect
        assert resp.status_code == 302

        # AND sample in case obj should have been updated
        updated_case = store.case_collection.find_one({"_id": cancer_case_obj["_id"]})
        updated_sample = updated_case["individuals"][0]

        assert updated_sample["tumor_purity"] != old_tumor_purity
        assert updated_sample["tumor_type"] != old_tumor_type
        assert updated_sample["tissue_type"] != old_tissue_type


def test_institutes(app):
    # GIVEN an initialized app
    # GIVEN a valid user

    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # WHEN accessing the institutes page
        resp = client.get(url_for("cases.index"))

        # THEN it should return a page
        assert resp.status_code == 200


def test_case(app, case_obj, institute_obj):
    # GIVEN an initialized app
    # GIVEN a valid user, case and institute

    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # WHEN accessing the case page
        resp = client.get(
            url_for(
                "cases.case",
                institute_id=institute_obj["internal_id"],
                case_name=case_obj["display_name"],
            )
        )

        # THEN it should return a page
        assert resp.status_code == 200


def test_case_sma(app, case_obj, institute_obj):
    # GIVEN an initialized app
    # GIVEN a valid user, case and institute

    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # WHEN accessing the case SMN CN page
        resp = client.get(
            url_for(
                "cases.sma",
                institute_id=institute_obj["internal_id"],
                case_name=case_obj["display_name"],
            )
        )

        # THEN it should return a page
        assert resp.status_code == 200


def test_update_individual(app, user_obj, institute_obj, case_obj):
    # GIVEN an initialized app
    # GIVEN a valid user and institute

    # And a case individual with no age (tissue type is default blood):
    case_obj = store.case_collection.find_one()
    assert case_obj["individuals"][0].get("age") is None
    assert case_obj["individuals"][0]["tissue_type"] == "blood"

    with app.test_client() as client:

        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # WHEN posting a request with info for updating one of the case samples:
        ind_id = case_obj["individuals"][0]["individual_id"]
        form_data = {
            "update_ind": ind_id,
            "_".join(["age", ind_id]): "2.5",
            "_".join(["tissue", ind_id]): "muscle",
        }

        resp = client.post(
            url_for(
                "cases.update_individual",
                institute_id=institute_obj["internal_id"],
                case_name=case_obj["display_name"],
            ),
            data=form_data,
        )

        # THEN the returned HTML page should redirect
        assert resp.status_code == 302

        # Then case obj should have been updated:
        updated_case = store.case_collection.find_one({"_id": case_obj["_id"]})
        updated_ind = updated_case["individuals"][0]
        assert updated_ind["individual_id"] == ind_id
        assert updated_ind["age"] == 2.5
        assert updated_ind["tissue_type"] == "muscle"

        # And an associated event should have been created in the database
        assert store.event_collection.find_one(
            {"case": updated_case["_id"], "verb": "update_individual"}
        )


def test_case_synopsis(app, institute_obj, case_obj):
    # GIVEN an initialized app
    # GIVEN a valid user and institute

    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        req_data = {"synopsis": "test synopsis"}

        # WHEN updating the synopsis of a case
        resp = client.post(
            url_for(
                "cases.case_synopsis",
                institute_id=institute_obj["internal_id"],
                case_name=case_obj["display_name"],
                data=req_data,
            )
        )
        # then it should return a redirected page
        assert resp.status_code == 302


def test_download_hpo_genes(app, case_obj, institute_obj):
    """Test the endpoint that downloads the dynamic gene list for a case"""

    # GIVEN a case containing a dynamic gene list
    dynamic_gene_list = [
        {"hgnc_symbol": "ACTA2", "hgnc_id": 130, "description": "actin alpha 2, smooth muscle"},
        {"hgnc_symbol": "LMNB2", "hgnc_id": 6638, "description": "lamin B2"},
    ]

    store.case_collection.find_one_and_update(
        {"_id": case_obj["_id"]}, {"$set": {"dynamic_gene_list": dynamic_gene_list}}
    )

    # GIVEN an initialized app
    # GIVEN a valid user and institute
    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))

        # WHEN the endpoint for downloading the case dynamic gene list is invoked
        resp = client.get(
            url_for(
                "cases.download_hpo_genes",
                institute_id=institute_obj["_id"],
                case_name=case_obj["display_name"],
            )
        )
        # THEN the response should be successful
        assert resp.status_code == 200
        # And should download a txt file
        assert resp.mimetype == "text/csv"


def test_case_report(app, institute_obj, case_obj):
    # Test the web page containing the general case report

    # GIVEN an initialized app and a valid user and institute
    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # When clicking on 'general' button on case page
        resp = client.get(
            url_for(
                "cases.case_report",
                institute_id=institute_obj["internal_id"],
                case_name=case_obj["display_name"],
            )
        )
        # a successful response should be returned
        assert resp.status_code == 200


def test_case_diagnosis(app, institute_obj, case_obj):
    # Test the web page containing the general case report

    # GIVEN an initialized app and a valid user and institute
    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        req_data = {"omim_term": "OMIM:615349"}

        # When updating an OMIM diagnosis for a case
        resp = client.post(
            url_for(
                "cases.case_diagnosis",
                institute_id=institute_obj["internal_id"],
                case_name=case_obj["display_name"],
            ),
            data=req_data,
        )
        # Response should be redirected to case page
        assert resp.status_code == 302


def test_pdf_case_report(app, institute_obj, case_obj):
    # Test the web page containing the general case report

    # GIVEN an initialized app and a valid user and institute
    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # When clicking on 'Download PDF' button on general report page
        resp = client.get(
            url_for(
                "cases.pdf_case_report",
                institute_id=institute_obj["internal_id"],
                case_name=case_obj["display_name"],
            )
        )
        # a successful response should be returned
        assert resp.status_code == 200


def test_mt_report(app, institute_obj, case_obj):
    # GIVEN an initialized app
    # GIVEN a valid user and institute

    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # When clicking on 'mtDNA report' on case page
        resp = client.get(
            url_for(
                "cases.mt_report",
                institute_id=institute_obj["internal_id"],
                case_name=case_obj["display_name"],
            )
        )
        # a successful response should be returned
        assert resp.status_code == 200
        # and it should contain a zipped file, not HTML code
        assert resp.mimetype == "application/zip"


def test_matchmaker_add(app, institute_obj, case_obj):
    # GIVEN an initialized app
    # GIVEN a valid user and institute

    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # WHEN accessing the case page
        resp = client.post(
            url_for(
                "cases.matchmaker_add",
                institute_id=institute_obj["internal_id"],
                case_name=case_obj["display_name"],
            )
        )
        # page redirects in the views anyway, so it will return a 302 code
        assert resp.status_code == 302


def test_matchmaker_matches(app, institute_obj, case_obj, mme_submission, user_obj, monkeypatch):

    # Given a case object with a MME submission
    case_obj["mme_submission"] = mme_submission
    store.update_case(case_obj)

    res = store.case_collection.find({"mme_submission": {"$exists": True}})
    assert sum(1 for i in res) == 1

    # Monkeypatch response with MME matches
    def mock_matches(*args, **kwargs):
        return {"institute": institute_obj, "case": case_obj, "matches": {}}

    monkeypatch.setattr(controllers, "mme_matches", mock_matches)

    # GIVEN an initialized app
    # GIVEN a valid institute and a user with mme_submitter role
    store.user_collection.update_one(
        {"_id": user_obj["_id"]}, {"$set": {"roles": ["mme_submitter"]}}
    )

    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # Given mock MME connection parameters
        current_app.config["MME_URL"] = "http://fakey_mme_url:fakey_port"
        current_app.config["MME_TOKEN"] = TEST_TOKEN

        # WHEN accessing the case page
        resp = client.get(
            url_for(
                "cases.matchmaker_matches",
                institute_id=institute_obj["internal_id"],
                case_name=case_obj["display_name"],
            )
        )

        # Then a successful response should be generated
        assert resp.status_code == 200


def test_matchmaker_match(app, institute_obj, case_obj, mme_submission, user_obj, monkeypatch):

    # Given a case object with a MME submission
    case_obj["mme_submission"] = mme_submission
    store.update_case(case_obj)

    res = store.case_collection.find({"mme_submission": {"$exists": True}})
    assert sum(1 for i in res) == 1

    # Monkeypatch response with MME match
    def mock_match(*args, **kwargs):
        return [{"status_code": 200}]

    monkeypatch.setattr(controllers, "mme_match", mock_match)

    # GIVEN an initialized app
    # GIVEN a valid institute and a user with mme_submitter role
    store.user_collection.update_one(
        {"_id": user_obj["_id"]}, {"$set": {"roles": ["mme_submitter"]}}
    )
    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # Given mock MME connection parameters
        current_app.config["MME_URL"] = "http://fakey_mme_url:fakey_port"
        current_app.config["MME_TOKEN"] = TEST_TOKEN

        # WHEN sending a POST request to match a patient
        resp = client.post(
            url_for(
                "cases.matchmaker_match",
                institute_id=institute_obj["internal_id"],
                case_name=case_obj["display_name"],
                target="mock_node_id",
            )
        )
        # page redirects in the views anyway, so it will return a 302 code
        assert resp.status_code == 302


def test_matchmaker_delete(app, institute_obj, case_obj, mme_submission):
    # GIVEN an initialized app
    # GIVEN a valid user and institute

    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # add MME submission to case object
        store.case_collection.find_one_and_update(
            {"_id": case_obj["_id"]}, {"$set": {"mme_submission": mme_submission}}
        )
        res = store.case_collection.find({"mme_submission": {"$exists": True}})
        assert sum(1 for i in res) == 1

        # WHEN accessing the case page
        resp = client.post(
            url_for(
                "cases.matchmaker_delete",
                institute_id=institute_obj["internal_id"],
                case_name=case_obj["display_name"],
            )
        )
        # page redirects in the views anyway, so it will return a 302 code
        assert resp.status_code == 302


def test_status(app, institute_obj, case_obj, user_obj):
    # GIVEN an initialized app
    # GIVEN a valid user and institute

    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # make sure test case status is inactive
        assert case_obj["status"] == "inactive"

        # use status view to update status for test case
        request_data = {"status": "prioritized"}
        resp = client.post(
            url_for(
                "cases.status",
                institute_id=institute_obj["internal_id"],
                case_name=case_obj["display_name"],
                params=request_data,
            )
        )

        assert resp.status_code == 302  # page should be redirected


def test_html_delivery_report(app, institute_obj, case_obj, user_obj):

    # GIVEN an initialized app
    # GIVEN a valid user and institute
    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # AND the case has a delivery report
        store.case_collection.update_one(
            {"_id": case_obj["_id"]}, {"$set": {"delivery_report": delivery_report_path}},
        )

        # WHEN accessing the delivery report page
        resp = client.get(
            url_for(
                "cases.delivery_report",
                institute_id=institute_obj["internal_id"],
                case_name=case_obj["display_name"],
            )
        )

        # THEN the endpoint should return the delivery report HTML page
        assert "Leveransrapport Clinical Genomics" in str(resp.data)


def test_pdf_delivery_report(app, institute_obj, case_obj, user_obj):

    # GIVEN an initialized app
    # GIVEN a valid user and institute
    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # AND the case has a delivery report
        store.case_collection.update_one(
            {"_id": case_obj["_id"]}, {"$set": {"delivery_report": delivery_report_path}},
        )

        # WHEN accessing the delivery report page with the format=pdf param
        resp = client.get(
            url_for(
                "cases.delivery_report",
                institute_id=institute_obj["internal_id"],
                case_name=case_obj["display_name"],
                format="pdf",
            )
        )

        # a successful response should be returned
        assert resp.status_code == 200
        # and it should contain a pdf file, not HTML code
        assert resp.mimetype == "application/pdf"


def test_omimterms(app, test_omim_term):
    """Test The API which returns all OMIM terms when queried from case page"""

    # GIVEN a database containing at least one OMIM term
    store.disease_term_collection.insert_one(test_omim_term)

    # GIVEN an initialized app
    # GIVEN a valid user and institute
    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # WHEN the API is invoked with a query string containing part of the OMIM term description
        resp = client.get(url_for("cases.omimterms", query="5-oxo"))
        # THEN it should return a valid response
        assert resp.status_code == 200

        # containing the OMIM term
        assert test_omim_term["_id"] in str(resp.data)

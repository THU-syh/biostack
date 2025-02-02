# -*- coding: utf-8 -*-
from flask import url_for
from scout.server.extensions import store


def test_overview(app, user_obj, institute_obj):
    # GIVEN an initialized app
    # GIVEN a valid user and institute

    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # WHEN accessing the cases page
        resp = client.get(url_for("overview.institutes"))

        # THEN it should return a page
        assert resp.status_code == 200


def test_institute(app, user_obj, institute_obj):
    """Test function that creates institute update form and updates an institute"""

    # insert 2 mock HPO terms in database, for later use
    mock_disease_terms = [
        {"_id": "HP:0001298", "description": "Encephalopathy", "hpo_id": "HP:0001298"},
        {"_id": "HP:0001250", "description": "Seizures", "hpo_id": "HP:0001250"},
    ]
    for term in mock_disease_terms:
        store.load_hpo_term(term)
        assert store.hpo_term(term["_id"])

    # GIVEN an initialized app
    # GIVEN a valid user and institute
    with app.test_client() as client:

        client.get(url_for("auto_login"))

        # WHEN accessing the cases page (GET method)
        resp = client.get(
            url_for("overview.institute_settings", institute_id=institute_obj["internal_id"])
        )

        # THEN it should return a page
        assert resp.status_code == 200

        # WHEN updating an institute using the following form
        form_data = {
            "display_name": "updated name",
            "sanger_emails": ["john@doe.com"],
            "coverage_cutoff": "15",
            "frequency_cutoff": "0.001",
            "cohorts": ["test cohort 1", "test cohort 2"],
            "institutes": ["cust111", "cust222"],
            "pheno_groups": [
                "HP:0001298 , Encephalopathy ( ENC )",
                "HP:0001250 , Seizures ( EP )",
            ],
        }

        # via POST request
        resp = client.post(
            url_for("overview.institute_settings", institute_id=institute_obj["internal_id"]),
            data=form_data,
        )
        assert resp.status_code == 200

        # THEN the institute object should be updated with the provided form data
        updated_institute = store.institute_collection.find_one()
        assert updated_institute["display_name"] == form_data["display_name"]
        assert updated_institute["sanger_recipients"] == form_data["sanger_emails"]
        assert updated_institute["coverage_cutoff"] == int(form_data["coverage_cutoff"])
        assert updated_institute["frequency_cutoff"] == float(form_data["frequency_cutoff"])
        assert updated_institute["cohorts"] == form_data["cohorts"]
        assert updated_institute["collaborators"] == form_data["institutes"]
        assert len(updated_institute["phenotype_groups"]) == 2  # one for each HPO term


def test_cases(app, institute_obj):
    # GIVEN an initialized app
    # GIVEN a valid user and institute

    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # WHEN accessing the cases page
        resp = client.get(url_for("overview.cases", institute_id=institute_obj["internal_id"]))

        # THEN it should return a page
        assert resp.status_code == 200

        # test query passing parameters in seach form
        request_data = {
            "limit": "100",
            "skip_assigned": "on",
            "is_research": "on",
            "query": "case_id",
        }
        resp = client.get(
            url_for(
                "overview.cases", institute_id=institute_obj["internal_id"], params=request_data,
            )
        )
        # response should return a page
        assert resp.status_code == 200

        sorting_options = ["analysis_date", "track", "status"]
        for option in sorting_options:
            # test query passing the sorting option to the cases view
            request_data = {"sort": option}
            resp = client.get(
                url_for(
                    "overview.cases",
                    institute_id=institute_obj["internal_id"],
                    params=request_data,
                )
            )
            # response should return a page
            assert resp.status_code == 200


def test_cases_query(app, case_obj, institute_obj):
    # GIVEN an initialized app
    # GIVEN a valid user and institute

    slice_query = case_obj["display_name"]

    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # WHEN accessing the cases page with a query
        resp = client.get(
            url_for("overview.cases", query=slice_query, institute_id=institute_obj["internal_id"],)
        )

        # THEN it should return a page
        assert resp.status_code == 200


def test_cases_panel_query(app, case_obj, parsed_panel, institute_obj):
    # GIVEN an initialized app
    # GIVEN a valid user and institute

    slice_query = parsed_panel["panel_id"]

    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # WHEN accessing the cases page with a query
        resp = client.get(
            url_for("overview.cases", query=slice_query, institute_id=institute_obj["internal_id"],)
        )

        # THEN it should return a page
        assert resp.status_code == 200


def test_causatives(app, user_obj, institute_obj, case_obj):
    # GIVEN an initialized app
    # GIVEN a valid user and institute
    # There should be no causative variants for test case:
    assert "causatives" not in case_obj
    var1_id = "4c7d5c70d955875504db72ef8e1abe77"  # in POT1 gene
    var2_id = "e24b65bf27feacec6a81c8e9e19bd5f1"  # in TBX1 gene
    var_ids = [var1_id, var2_id]

    # for each variant
    for var_id in var_ids:
        # update case by marking variant as causative:
        variant_obj = store.variant(document_id=var_id)
        store.mark_causative(
            institute=institute_obj,
            case=case_obj,
            user=user_obj,
            link="causative_var_link/{}".format(variant_obj["_id"]),
            variant=variant_obj,
        )
    updated_case = store.case_collection.find_one({"_id": case_obj["_id"]})
    # The above variants should be registered as causatives in case object
    assert updated_case["causatives"] == var_ids

    # Call scout causatives view and check if the above causatives are displayed
    with app.test_client() as client:

        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # WHEN accessing the case page
        resp = client.get(url_for("overview.causatives", institute_id=institute_obj["internal_id"]))

        # THEN it should return a page
        assert resp.status_code == 200
        # with variant 1
        assert var1_id in str(resp.data)
        # and variant 2
        assert var2_id in str(resp.data)

        # Filter causatives by gene (POT1)
        resp = client.get(
            url_for(
                "overview.causatives",
                institute_id=institute_obj["internal_id"],
                query="17284 | POT1 (DKFZp586D211, hPot1, POT1)",
            )
        )
        # THEN it should return a page
        assert resp.status_code == 200
        # with variant 1
        assert var1_id in str(resp.data)
        # but NOT variant 2
        assert var2_id not in str(resp.data)


def test_gene_variants_filter(app, institute_obj, case_obj):
    """Test the function that allows searching SNVs and INDELS using filters"""

    # GIVEN an initialized app
    # GIVEN a valid user and institute
    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # When user submits a query for a variants in a specific gene and a rank score
        filter_query = {
            "hgnc_symbols": "POT1",
            "variant_type": ["clinical"],
            "rank_score": 11,
        }

        resp = client.post(
            url_for("overview.gene_variants", institute_id=institute_obj["internal_id"]),
            data=filter_query,
        )
        # THEN it should return a page
        assert resp.status_code == 200

        # containing  variants in that genes
        assert "POT1" in str(resp.data)


def test_institute_users(app, institute_obj, user_obj):
    """Test the link to all institute users"""
    # GIVEN an initialized app
    # GIVEN a valid user and institute
    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))

        # WHEN accessing the cases page
        resp = client.get(
            url_for("overview.institute_users", institute_id=institute_obj["internal_id"])
        )

        # THEN it should return a page
        assert resp.status_code == 200

        # Containing the test user's name
        assert user_obj["name"] in str(resp.data)


def test_clinvar_submissions(app, institute_obj, clinvar_variant, clinvar_casedata):
    """"Test the web page containing the clinvar submissions for an institute"""

    # GIVEN an institute with a clinvar submission
    store.create_submission(institute_obj["_id"])
    open_submission = store.get_open_clinvar_submission(institute_obj["_id"])
    submission_with_data = store.add_to_submission(
        open_submission["_id"], ([clinvar_variant], [clinvar_casedata])
    )
    assert submission_with_data

    # GIVEN an initialized app and a valid user and institute
    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))
        assert resp.status_code == 200

        # When visiting the clinvar submission page (get request)
        resp = client.get(
            url_for("overview.clinvar_submissions", institute_id=institute_obj["internal_id"],)
        )

        # a successful response should be returned
        assert resp.status_code == 200
        assert str(submission_with_data["_id"]) in str(resp.data)


def test_rename_clinvar_samples(app, institute_obj, clinvar_variant, clinvar_casedata):
    """Test the form button triggering the renaming of samples for a clinvar submission"""

    # GIVEN an institute with a clinvar submission
    store.create_submission(institute_obj["_id"])
    open_submission = store.get_open_clinvar_submission(institute_obj["_id"])
    submission_with_data = store.add_to_submission(
        open_submission["_id"], ([clinvar_variant], [clinvar_casedata])
    )
    assert submission_with_data["_id"]

    # GIVEN an initialized app and a valid user
    with app.test_client() as client:
        # GIVEN that the user could be logged in
        resp = client.get(url_for("auto_login"))

        case_id = clinvar_casedata["case_id"]
        old_name = clinvar_casedata["individual_id"]

        form_data = dict(new_name="new_sample_name",)

        # WHEN the sample name is edited from the submission page (POST request)
        resp = client.post(
            url_for(
                f"overview.clinvar_rename_casedata",
                submission=submission_with_data["_id"],
                case=case_id,
                old_name=old_name,
            ),
            data=form_data,
        )
        # a successful response should be redirect to the submssions page
        assert resp.status_code == 302

        # And the sample name should have been updated in the database
        updated_casedata = store.clinvar_collection.find_one({"_id": clinvar_casedata["_id"]})
        assert updated_casedata["individual_id"] != clinvar_casedata["individual_id"]

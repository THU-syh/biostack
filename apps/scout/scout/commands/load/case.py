# -*- coding: utf-8 -*-
import logging

from pprint import pprint as pp

from flask.cli import with_appcontext
import click
import yaml
import traceback

from cyvcf2 import VCF

from scout.load import load_scout
from scout.parse.case import parse_case_data
from scout.exceptions import IntegrityError, ConfigError
from scout.server.extensions import store

LOG = logging.getLogger(__name__)


@click.command("case", short_help="Load a case")
@click.option("--vcf", type=click.Path(exists=True), help="path to clinical VCF file to be loaded")
@click.option(
    "--vcf-sv", type=click.Path(exists=True), help="path to clinical SV VCF file to be loaded",
)
@click.option(
    "--vcf-cancer",
    type=click.Path(exists=True),
    help="path to clinical cancer VCF file to be loaded",
)
@click.option(
    "--vcf-cancer-sv",
    type=click.Path(exists=True),
    help="path to clinical cancer SV VCF file to be loaded",
)
@click.option(
    "--vcf-str", type=click.Path(exists=True), help="path to clinical STR VCF file to be loaded",
)
@click.option("--owner", help="parent institute for the case", default="test")
@click.option("--ped", type=click.File("r"))
@click.option("-u", "--update", is_flag=True)
@click.option(
    "--keep-actions/--no-keep-actions",
    default=True,
    help="Transfer user actions from old variants when updating.",
)
@click.option("--no-variants", is_flag=False)
@click.argument("config", type=click.File("r"), required=False)
@click.option("--peddy-ped", type=click.Path(exists=True), help="path to a peddy.ped file")
@click.option("--peddy-sex", type=click.Path(exists=True), help="path to a sex_check.csv file")
@click.option("--peddy-check", type=click.Path(exists=True), help="path to a ped_check.csv file")
@with_appcontext
def case(
    vcf,
    vcf_sv,
    vcf_cancer,
    vcf_cancer_sv,
    vcf_str,
    owner,
    ped,
    update,
    config,
    no_variants,
    peddy_ped,
    peddy_sex,
    peddy_check,
    keep_actions,
):
    """Load a case into the database.

    A case can be loaded without specifying vcf files and/or bam files
    """
    adapter = store

    if config is None and ped is None:
        LOG.warning("Please provide either scout config or ped file")
        raise click.Abort()

    # Scout needs a config object with the neccessary information
    # If no config is used create a dictionary
    config_raw = yaml.load(config, Loader=yaml.FullLoader) if config else {}

    try:
        config_data = parse_case_data(
            config=config_raw,
            ped=ped,
            owner=owner,
            vcf_snv=vcf,
            vcf_sv=vcf_sv,
            vcf_str=vcf_str,
            vcf_cancer=vcf_cancer,
            vcf_cancer_sv=vcf_cancer_sv,
            peddy_ped=peddy_ped,
            peddy_sex=peddy_sex,
            peddy_check=peddy_check,
        )
    except SyntaxError as err:
        LOG.warning(err)
        raise click.Abort()
    except KeyError as err:
        LOG.error("KEYERROR {} missing when loading '{}'".format(err, config.name))
        LOG.debug("Stack trace: {}".format(traceback.format_exc()))
        raise click.Abort()

    LOG.info("Use family %s" % config_data["family"])

    try:
        case_obj = adapter.load_case(config_data, update, keep_actions)
    except Exception as err:
        LOG.error("Something went wrong during loading")
        LOG.warning(err)
        raise click.Abort()

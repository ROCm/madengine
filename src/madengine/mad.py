#!/usr/bin/env python3
"""MAD Engine CLI tool.

This script provides a command-line interface to run models, generate reports, and tools for profiling and tracing.
This tool is used to run LLMs and Deep Learning models locally.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import argparse
import logging

import sys
# MAD Engine imports
from madengine import __version__
from madengine.tools.run_models import RunModels
from madengine.tools.discover_models import DiscoverModels
from madengine.tools.create_table_db import CreateTable
from madengine.tools.update_table_db import UpdateTable
from madengine.tools.upload_mongodb import MongoDBHandler
from madengine.tools.update_perf_csv import UpdatePerfCsv
from madengine.tools.csv_to_html import ConvertCsvToHtml
from madengine.tools.csv_to_email import ConvertCsvToEmail
from madengine.core.constants import MODEL_DIR  # pylint: disable=unused-import
from madengine.utils.gpu_validator import validate_gpu_installation, GPUInstallationError, detect_gpu_vendor, GPUVendor

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Sub-command functions
# -----------------------------------------------------------------------------
# Router of the command-line arguments to the corresponding functions
def run_models(args: argparse.Namespace):
    """Run models on container.

    Args:
        args: The command-line arguments.
    """
    logger.info("Running models on container")
    run_models_instance = RunModels(args=args)
    return run_models_instance.run()


def discover_models(args: argparse.Namespace):
    """Discover the models.

    Args:
        args: The command-line arguments.
    """
    logger.info("Discovering all models in the project")
    discover_models_instance = DiscoverModels(args=args)
    return discover_models_instance.run()


def update_perf_csv(args):
    """Update performance metrics of models perf.csv to database.

    Args:
        args: The command-line arguments.
    """
    logger.info("Running update_perf_csv")
    update_perf_csv_instance = UpdatePerfCsv(args=args)
    return update_perf_csv_instance.run()


def csv_to_html(args):
    """Convert CSV to HTML report of models.

    Args:
        args: The command-line arguments.
    """
    logger.info("Running csv_to_html")
    convert_csv_to_html = ConvertCsvToHtml(args=args)
    return convert_csv_to_html.run()


def csv_to_email(args):
    """Convert CSV to Email of models.

    Args:
        args: The command-line arguments.
    """
    logger.info("Convert CSV to Email of models")
    convert_csv_to_email = ConvertCsvToEmail(args=args)
    return convert_csv_to_email.run()


def create_table(args):
    """Create table in DB.

    Args:
        args: The command-line arguments.
    """
    logger.info("Create table in DB")
    create_table_instance = CreateTable(args=args)
    return create_table_instance.run()


def update_table(args):
    """Update table in DB.

    Args:
        args: The command-line arguments.
    """
    logger.info("Update table in DB")
    update_table_instance = UpdateTable(args=args)
    return update_table_instance.run()


def upload_mongodb(args):
    """Upload to MongoDB.

    Args:
        args: The command-line arguments.
    """   
    print(f"Uploading to MongoDB")    
    upload_mongodb = MongoDBHandler(args=args)
    return upload_mongodb.run()


def validate_gpu(args):
    """Validate GPU installation (ROCm for AMD, CUDA for NVIDIA).
    
    Args:
        args: The command-line arguments.
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    verbose = args.verbose if hasattr(args, 'verbose') else False
    
    try:
        # Detect GPU vendor and run appropriate validation
        result = validate_gpu_installation(vendor=None, verbose=verbose, raise_on_error=False)
        
        # Print summary based on validation result
        if result.is_valid:
            print()
            print("=" * 70)
            print(f"✓ {result.vendor.value} GPU Installation is VALID")
            print("=" * 70)
            if result.version:
                version_label = "ROCm Version" if result.vendor == GPUVendor.AMD else "Driver/CUDA Version"
                print(f"{version_label}: {result.version}")
            print()
            print("You can proceed with running madengine workloads:")
            print("  madengine run --tags <your_tags>")
            print()
            return 0
        else:
            print()
            print("=" * 70)
            print(f"✗ {result.vendor.value} GPU Installation Validation FAILED")
            print("=" * 70)
            print()
            
            if result.issues:
                print("Critical Issues:")
                for issue in result.issues:
                    print(f"  - {issue}")
                print()
            
            if result.warnings:
                print("Warnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
                print()
            
            if result.suggestions:
                print("Suggested Actions:")
                for suggestion in result.suggestions:
                    print(f"  • {suggestion}")
                print()
            
            print("Please fix the issues above before running madengine workloads.")
            print()
            return 1
            
    except GPUInstallationError as e:
        print()
        print("=" * 70)
        print("GPU Installation Validation FAILED")
        print("=" * 70)
        print()
        print(str(e))
        print()
        return 1
    except Exception as e:
        print(f"✗ Unexpected error during validation: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return 1


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main():
    """Main function to parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="A Models automation and dashboarding command-line tool to run LLMs and Deep Learning models locally."
    )

    parser.add_argument("-v", "--version", action="version", version=__version__)

    subparsers = parser.add_subparsers(
        title="Commands",
        description="Available commands for running models, generating reports, and toolings.",
        dest="command",
    )

    # Run models command
    parser_run = subparsers.add_parser(
        "run",
        description="Run LLMs and Deep Learning models on container",
        help="Run models on container",
    )
    parser_run.add_argument(
        "--tags", nargs="+", default=[], help="tags to run (can be multiple)."
    )

    # Deprecated Tag
    parser_run.add_argument(
        "--ignore-deprecated-flag",
        action="store_true",
        help="Force run deprecated models even if marked deprecated.",
    )

    parser_run.add_argument(
        "--timeout",
        type=int,
        default=-1,
        help="time out for model run in seconds; Overrides per-model timeout if specified or default timeout of 7200 (2 hrs).\
                                               Timeout of 0 will never timeout.",
    )
    parser_run.add_argument(
        "--live-output",
        action="store_true",
        help="prints output in real-time directly on STDOUT",
    )
    parser_run.add_argument(
        "--clean-docker-cache",
        action="store_true",
        help="rebuild docker image without using cache",
    )
    parser_run.add_argument(
        "--additional-context-file",
        default=None,
        help="additonal context, as json file, to filter behavior of workloads. Overrides detected contexts.",
    )
    parser_run.add_argument(
        "--additional-context",
        default="{}",
        help="additional context, as string representation of python dict, to filter behavior of workloads. "
        + " Overrides detected contexts and additional-context-file.",
    )
    parser_run.add_argument(
        "--data-config-file-name",
        default="data.json",
        help="custom data configuration file.",
    )
    parser_run.add_argument(
        "--tools-json-file-name",
        default="./scripts/common/tools.json",
        help="custom tools json configuration file.",
    )
    parser_run.add_argument(
        "--generate-sys-env-details",
        default=True,
        help="generate system config env details by default",
    )
    parser_run.add_argument(
        "--force-mirror-local",
        default=None,
        help="Path to force all relevant dataproviders to mirror data locally on.",
    )
    parser_run.add_argument(
        "--keep-alive",
        action="store_true",
        help="keep Docker container alive after run; will keep model directory after run",
    )
    parser_run.add_argument(
        "--keep-model-dir", action="store_true", help="keep model directory after run"
    )
    parser_run.add_argument(
        "--skip-model-run",
        action="store_true",
        help="skips running the model; will not keep model directory after run unless specified through keep-alive or keep-model-dir",
    )
    parser_run.add_argument(
        "--disable-skip-gpu-arch",
        action="store_true",
        help="disables skipping model based on gpu architecture",
    )
    parser_run.add_argument("-o", "--output", default="perf.csv", help="output file")
    parser_run.set_defaults(func=run_models)

    # Discover models command
    parser_discover = subparsers.add_parser(
        "discover",
        description="Discover all models in the project",
        help="Discover the models.",
    )
    parser_discover.add_argument(
        "--tags",
        nargs="+",
        default=[],
        help="tags to discover models (can be multiple).",
    )
    parser_discover.set_defaults(func=discover_models)

    # Report command
    parser_report = subparsers.add_parser(
        "report", description="", help="Generate report of models"
    )
    subparsers_report = parser_report.add_subparsers(
        title="Report Commands",
        description="Available commands for generating reports.",
        dest="report_command",
    )
    # Report subcommand update-perf
    parser_report_update_perf = subparsers_report.add_parser(
        "update-perf",
        description="Update performance metrics of models perf.csv to database.",
        help="Update perf.csv to database",
    )
    parser_report_update_perf.add_argument(
        "--single_result", help="path to the single result json"
    )
    parser_report_update_perf.add_argument(
        "--exception-result", help="path to the single result json"
    )
    parser_report_update_perf.add_argument(
        "--failed-result", help="path to the single result json"
    )
    parser_report_update_perf.add_argument(
        "--multiple-results", help="path to the results csv"
    )
    parser_report_update_perf.add_argument("--perf-csv", default="perf.csv")
    parser_report_update_perf.add_argument("--model-name")
    parser_report_update_perf.add_argument("--common-info")
    parser_report_update_perf.set_defaults(func=update_perf_csv)
    # Report subcommand to-html
    parser_report_html = subparsers_report.add_parser(
        "to-html",
        description="Convert CSV to HTML report of models.",
        help="Convert CSV to HTML report of models",
    )
    parser_report_html.add_argument("--csv-file-path", type=str)
    parser_report_html.set_defaults(func=csv_to_html)
    # Report subcommand to-email
    parser_report_email = subparsers_report.add_parser(
        "to-email",
        description="Convert CSV to Email of models.",
        help="Convert CSV to Email of models",
    )
    parser_report_email.add_argument(
        "--csv-file-path",
        type=str,
        default=".",
        help="Path to the directory containing the CSV files.",
    )
    parser_report_email.set_defaults(func=csv_to_email)

    # Database command
    parser_database = subparsers.add_parser("database", help="CRUD for database")
    subparsers_database = parser_database.add_subparsers(
        title="Database Commands",
        description="Available commands for database, such as creating and updating table in DB.",
        dest="database_command",
    )
    # Database subcommand creating tabe
    parser_database_create_table = subparsers_database.add_parser(
        "create-table", description="Create table in DB.", help="Create table in DB"
    )
    parser_database_create_table.add_argument(
        "-v", "--verbose", action="store_true", help="verbose output"
    )
    parser_database_create_table.set_defaults(func=create_table)
    # Database subcommand updating table
    parser_database_update_table = subparsers_database.add_parser(
        "update-table", description="Update table in DB.", help="Update table in DB"
    )
    parser_database_update_table.add_argument(
        "--csv-file-path", type=str, help="Path to the csv file"
    )
    parser_database_update_table.add_argument(
        "--model-json-path", type=str, help="Path to the model json file"
    )
    parser_database_update_table.set_defaults(func=update_table)
    # Database subcommand uploading to MongoDB
    parser_database_upload_mongodb = subparsers_database.add_parser(
        "upload-mongodb", description="Update table in DB.", help="Update table in DB"
    )
    parser_database_upload_mongodb.add_argument(
        "--csv-file-path",
        type=str,
        default="perf_entry.csv",
        help="Path to the csv file",
    )
    parser_database_upload_mongodb.add_argument(
        "--database-name", type=str, required=True, help="Name of the MongoDB database"
    )
    parser_database_upload_mongodb.add_argument(
        "--collection-name",
        type=str,
        required=True,
        help="Name of the MongoDB collection",
    )
    parser_database_upload_mongodb.set_defaults(func=upload_mongodb)
    
    # Validate GPU command
    parser_validate = subparsers.add_parser('validate', description="Validate GPU installation (ROCm for AMD, CUDA for NVIDIA)", help='Validate GPU installation')
    parser_validate.add_argument('-v', '--verbose', action='store_true', help='Show detailed validation output')
    parser_validate.set_defaults(func=validate_gpu)
    
    args = parser.parse_args()

    if args.command:
        result = args.func(args)
        if args.command == 'validate' and result is not None:
            sys.exit(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

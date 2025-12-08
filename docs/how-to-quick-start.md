# Quickstart

Run madengine CLI on your local machine.

```shell
(venv) test-node:~/MAD$ madengine --help
usage: madengine [-h] [-v] {run,discover,report,database} ...

A Models automation and dashboarding command-line tool to run LLMs and Deep Learning models locally.

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit

Commands:
  Available commands for running models, generating reports, and toolings.

  {run,discover,report,database}
    run                 Run models on container
    discover            Discover the models.
    report              Generate report of models
    database            CRUD for database
```

## Run models

You can use `madengine run` to benchmark the training and inference performance of various LLM and Deep Learning models/frameworks listed in [MAD](https://github.com/ROCm/MAD).

```shell
(venv) test-node:~/MAD$ madengine run --help
usage: madengine run [-h] [--tags TAGS [TAGS ...]] [--timeout TIMEOUT] [--live-output] [--clean-docker-cache] [--additional-context-file ADDITIONAL_CONTEXT_FILE]
                     [--additional-context ADDITIONAL_CONTEXT] [--data-config-file-name DATA_CONFIG_FILE_NAME] [--tools-json-file-name TOOLS_JSON_FILE_NAME]
                     [--generate-sys-env-details GENERATE_SYS_ENV_DETAILS] [--force-mirror-local FORCE_MIRROR_LOCAL] [--keep-alive] [--keep-model-dir]
                     [--disable-skip-gpu-arch] [-o OUTPUT]

Run LLMs and Deep Learning models on container

optional arguments:
  -h, --help            show this help message and exit
  --tags TAGS [TAGS ...]
                        tags to run (can be multiple).
  --timeout TIMEOUT     time out for model run in seconds; Overrides per-model timeout if specified or default timeout of 7200 (2 hrs). Timeout
                        of 0 will never timeout.
  --live-output          prints output in real-time directly on STDOUT
  --clean-docker-cache    rebuild docker image without using cache
  --additional-context-file ADDITIONAL_CONTEXT_FILE
                        additonal context, as json file, to filter behavior of workloads. Overrides detected contexts.
  --additional-context ADDITIONAL_CONTEXT
                        additional context, as string representation of python dict, to filter behavior of workloads. Overrides detected contexts and additional-
                        context-file.
  --data-config-file-name DATA_CONFIG_FILE_NAME
                        custom data configuration file.
  --tools-json-file-name TOOLS_JSON_FILE_NAME
                        custom tools json configuration file.
  --generate-sys-env-details GENERATE_SYS_ENV_DETAILS
                        generate system config env details by default
  --force-mirror-local FORCE_MIRROR_LOCAL
                        Path to force all relevant dataproviders to mirror data locally on.
  --keep-alive          keep Docker container alive after run; will keep model directory after run
  --keep-model-dir      keep model directory after run
  --disable-skip-gpu-arch
                        disables skipping model based on gpu architecture
  -o OUTPUT, --output OUTPUT
                        output file
```

A CLI example to run a model (See pyt_huggingface_bert in https://github.com/ROCm/MAD/models.json):

```shell
madengine run --tags pyt_huggingface_bert --live-output --additional-context "{'guest_os': 'UBUNTU'}"
```

## Generate perf reports

Commands for generating reports.

```shell
(venv) test-node:~/MAD$ madengine report --help
usage: madengine report [-h] {update-perf,to-html,to-email} ...
 
optional arguments:
  -h, --help            show this help message and exit
 
Report Commands:
  Available commands for generating reports.
 
  {update-perf,to-html,to-email}
    update-perf         Update perf.csv to database
    to-html             Convert CSV to HTML report of models
    to-email            Convert CSV to Email of models
```

## Database

Commands for database, such as create and update table of DB.

```shell
(venv) test-node:~/MAD$ madengine database --help
usage: madengine database [-h] {create-table,update-table,upload-mongodb} ...

optional arguments:
  -h, --help            show this help message and exit

Database Commands:
  Available commands for database, such as creating and updating table in DB.

  {create-table,update-table,upload-mongodb}
    create-table        Create table in DB
    update-table        Update table in DB
    upload-mongodb      Update table in DB
```

## Tools in madengine

There are some additional tools packaged with madengine. They work with madengine CLI to profile GPU usage and get trace of ROCm libraries.

An example of profiling GPU usage with [rocprof](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/).

```shell
madengine run --tags pyt_huggingface_bert --additional-context "{'guest_os': 'UBUNTU','tools': [{'name':'rocprof'}]}"
```

An example of tracing library usage with [rocblas](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/logging.html).

```shell
madengine run --tags pyt_huggingface_bert --additional-context "{'guest_os': 'UBUNTU','tools': [{'name':'rocblas_trace'}]}"
```
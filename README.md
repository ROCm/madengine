# madengine
Set of interfaces to run various AI models from public MAD.

# What is madengine?

An AI Models automation and dashboarding command-line tool to run LLMs and Deep Learning models locally or remotely with CI. 

The madengine library is to support AI automation having following features:
- AI Models run reliably on supported platforms and drive software quality
- Simple, minimalistic, out-of-the-box solution that enables confidence on hardware and software stack
- Real-time, audience-relevant AI Models performance metrics tracking, presented in clear, intuitive manner
- Best practices for handling internal projects and external open-source projects

# Installation

madengine is meant to be used in conjunction with [MAD](https://github.com/ROCm/MAD). Below are the steps to set it up and run it using the command-line interface (CLI).

## Clone MAD
```
git clone git@github.com:ROCm/MAD.git
cd MAD
```

## Install madengine

### Install from source

```
# Create virtual environment if necessary
python3 -m venv venv

# Activate the virtual environment venv
source venv/bin/activate

# Clone madengine
git clone git@github.com:ROCm/madengine.git

# Change current working directory to madengine
cd madengine

# Install madengine from source:
pip install .

```

### Install from repo

You can also install the madengine library directly from the Github repository.

```
pip install git+https://github.com/ROCm/madengine.git@main
```

# Run madengine CLI

How to run madengine CLI on your local machine.

```shell
(venv) test-node:~/MAD$ madengine --help
usage: madengine [-h] [-v] {run,discover,report,database} ...

A Model automation and dashboarding command-line tool to run LLMs and Deep Learning models locally.

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit

Commands:
  Available commands for running models, generating reports, and toolings.

  {run,discover,report,database}
    run                 Run models on container
    discover            Discover the models
    report              Generate report of models
    database            CRUD for database
```

## Run models locally

Command to run LLMs and deep learning models in a container.

```
# An example CLI command to run a model
madengine run --tags pyt_huggingface_bert --live-output --additional-context "{'guest_os': 'UBUNTU'}"
```

```shell
(venv) test-node:~/MAD$ madengine run --help
usage: madengine run [-h] [--tags TAGS [TAGS ...]] [--timeout TIMEOUT] [--live-output] [--clean-docker-cache] [--additional-context-file ADDITIONAL_CONTEXT_FILE]
                     [--additional-context ADDITIONAL_CONTEXT] [--data-config-file-name DATA_CONFIG_FILE_NAME] [--tools-json-file-name TOOLS_JSON_FILE_NAME]
                     [--generate-sys-env-details GENERATE_SYS_ENV_DETAILS] [--force-mirror-local FORCE_MIRROR_LOCAL] [--keep-alive] [--keep-model-dir]
                     [--skip-model-run] [--disable-skip-gpu-arch] [-o OUTPUT]

Run LLMs and Deep Learning models on container

optional arguments:
  -h, --help            show this help message and exit
  --tags TAGS [TAGS ...]
                        tags to run (space-separated).
  --timeout TIMEOUT     time out for model run in seconds; Overrides per-model timeout if specified or default timeout of 7200 (2 hrs). Timeout of 0 will never
                        timeout.
  --live-output         prints output in real-time directly on STDOUT
  --clean-docker-cache  rebuild docker image without using cache
  --additional-context-file ADDITIONAL_CONTEXT_FILE
                        additional context, as json file, to filter behavior of workloads. Overrides detected contexts.
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
  --keep-alive          keep Docker container alive after run; keeps model directory after run
  --keep-model-dir      keep model directory after run
  --skip-model-run      skips running the model; will not keep model directory after run unless specified through keep-alive or keep-model-dir
  --disable-skip-gpu-arch
                        disables skipping model based on GPU architecture
  -o OUTPUT, --output OUTPUT
                        output file
```

For each model in models.json, the script:
- builds Docker images associated with each model. The images are named 'ci-$(model_name)', and are not removed after the script completes.
- starts the Docker container, with name, 'container_$(model_name)'. The container should automatically be stopped and removed whenever the script exits. 
- clones the git 'url', and runs the 'script' 
- generates the final perf.csv and perf.html

### Tag functionality for running model

The tag functionality enables users to select and run specific models whose tags match those provided with the `--tags` argument. Multiple tags can be specified in a space-separated format, and any model matching at least one tag will be selected (logical OR). Each model name in `models.json` is automatically available as a tag.

#### Search models with tags

Use cases of running models with static and dynamic search. Tags option supports searching models in models.json, scripts/model_dir/models.json, and scripts/model_dir/get_models_json.py. A user can add new models not only to the models.json file of DLM but also to the model folder in Flexible. To do this, the user needs to follow these steps:

Update models.json: Add the new model's configuration details to the models.json file. This includes specifying the model's name, version, and any other relevant metadata.
Place Model Files: Copy the model files into the appropriate directory within the model folder in Flexible. Ensure that the folder structure and file naming conventions match the expected format.

```
# 1. run models in ~/MAD/models.json
(venv) test-node:~/MAD$ madengine run --tags dummy --live-output

# 2. run model in ~/MAD/scripts/dummy2/models.json
(venv) test-node:~/MAD$ madengine run --tags dummy2:dummy_2 --live-output

# 3. run model in ~/MAD/scripts/dummy3/get_models_json.py
(venv) test-node:~/MAD$ madengine run --tags dummy3:dummy_3 --live-output

# 4. run model with configurations
(venv) test-node:~/MAD$ madengine run --tags dummy2:dummy_2:batch_size=512:in=32:out=16 --live-output

# 5. run model with configurations
(venv) test-node:~/MAD$ madengine run --tags dummy3:dummy_3:batch_size=512:in=32:out=16 --live-output
```

The configs of batch_size512:in32:out16 will be pass to environment variables and build arguments of Docker.

### Custom timeouts
The default timeout for model run is 2 hours. This can be overridden if the model in models.json contains a `'timeout' : TIMEOUT` entry. Both the default timeout and/or timeout specified in models.json can be overridden using `--timeout TIMEOUT` command-line argument. Setting `TIMEOUT` to 0 means that the model run will never time out.

### Live output functionality 
By default, `madengine` is silent. The output is piped to log files. By specifying `--live-output`, the output is printed in real-time to STDOUT. 

### Contexts
Contexts are run-time parameters that change how the model is executed. Some contexts are auto-detected. Detected contexts may be overridden. Contexts are also used to filter Dockerfile used in model.  

For more details, see [How to provide contexts](docs/how-to-provide-contexts.md).

### Credentials
Credentials to clone model git URLs are provided in a centralized `credential.json` file. Models that require special credentials for cloning have a special `cred` field in the model definition in `models.json`. This field denotes the specific credential in `credential.json` to use. Public models repositories can skip the `cred` field. 

There are several types of credentials supported. 

1. For HTTP/HTTPS git URLs, `username` and `password` should be provided in the credential. For Source Code Management (SCM) systems that support Access Tokens, the token can be substituted for the `password` field. The `username` and `password` will be passed as a Docker build argument and a container environment variable in the Docker build and run steps. For example, for `"cred":"AMD_GITHUB"` field in `models.json` and entry `"AMD_GITHUB": { "username": "github_username", "password":"pass" }` in `credential.json` the following Docker build arguments and container environment variables will be added: `AMD_GITHUB_USERNAME="github_username"` and `AMD_GITHUB_PASSWORD="pass"`. 
      
2. For SSH git URLs, `username` and `ssh_key_file` should be provided in the credential. The `username` is the SSH username, and `ssh_key_file` is the private ssh key, that has been registered with the SCM system. 
Due to legal requirements, credentials to access all models are not provided by default in DLM. Please contact the model owner if you wish to access and run the model. 

3. For NAS URLs, `HOST`, `PORT`, `USERNAME`, and `PASSWORD` should be provided in the credential. Please check env variables starting with NAS in [Environment Variables](https://github.com/ROCm/madengine/blob/main/README.md#environment-variables)

3. For AWS S3 URLs, `USERNAME`, and `PASSWORD` should be provided in the credential with variable name as MAD_AWS_S3 as mentioned in [Environment Variables](https://github.com/ROCm/madengine/blob/main/README.md#environment-variables)


### Local data provider
The DLM user may wish to run a model locally multiple times, with the input data downloaded once, and reused subsequently. This functionality is only supported on models that support the Data Provider functionality. That is, the model specification in `models.json` have the `data` field, which points to a data specification in `data.json`.

To use existing data on a local path, add to the data specification, using a `local` field within `data.json`. By default, this path is mounted read-only. To change this path to read-write, specify the `readwrite` field to `'true'` in the data configuration.

If no data exists in local path, a local copy of data can be downloaded using by setting the `mirrorlocal` field in data specification in `data.json`. Not all providers support `mirrorlocal`. For the ones that do support this feature, the remote data is mirrored on this host path during the first run. In subsequent runs, the data may be reused through synchronization mechanisms. If the user wishes to skip the remote synchronization, the same location can be set as a `local` data provider in data.json, with higher precedence, or as the only provider for the data, by locally editing `data.json`. 

Alternatively, the command-line argument, `--force-mirror-local` forces local mirroring on *all* workloads, to the provided FORCEMIRRORLOCAL path.

## Discover models

Commands for discovering models through models.json, scripts/{model_dir}/models.json, or scripts/{model_dir}/get_models_json.py

```
(venv) test-node:~/MAD$ madengine discover --help
usage: madengine discover [-h] [--tags TAGS [TAGS ...]]

Discover the models

optional arguments:
  -h, --help            show this help message and exit
  --tags TAGS [TAGS ...]
                        tags to discover models (can be multiple).
```

Use cases about how to discover models:

```
# 1 discover all models in DLM
(venv) test-node:~/MAD$ madengine discover  

# 2. discover specified model using tags in models.json of DLM
(venv) test-node:~/MAD$ madengine discover --tags dummy

# 3. discover specified model using tags in scripts/{model_dir}/models.json with static search i.e. models.json
(venv) test-node:~/MAD$ madengine discover --tags dummy2/dummy_2

# 4. discover specified model using tags in scripts/{model_dir}/get_models_json.py with dynamic search i.e. get_models_json.py
(venv) test-node:~/MAD$ madengine discover --tags dummy3/dummy_3

# 5. pass additional args to your model script from CLI
(venv) test-node:~/MAD$ madengine discover --tags dummy3/dummy_3:bs16

# 6. get multiple models using tags
(venv) test-node:~/MAD$ madengine discover --tags pyt_huggingface_bert pyt_huggingface_gpt2
```

Note: You cannot use a backslash '/' or a colon ':' in a model name or a tag for a model in `models.json` or `get_models_json.py`

## Generate reports

Commands for generating reports.

```
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

### Report command - Update perf CSV to database

Update perf.csv to database

```
(venv) test-node:~/MAD$ madengine report update-perf --help
usage: madengine report update-perf [-h] [--single_result SINGLE_RESULT] [--exception-result EXCEPTION_RESULT] [--failed-result FAILED_RESULT]
                                    [--multiple-results MULTIPLE_RESULTS] [--perf-csv PERF_CSV] [--model-name MODEL_NAME] [--common-info COMMON_INFO]

Update performance metrics of models perf.csv to database.

optional arguments:
  -h, --help            show this help message and exit
  --single_result SINGLE_RESULT
                        path to the single result json
  --exception-result EXCEPTION_RESULT
                        path to the single result json
  --failed-result FAILED_RESULT
                        path to the single result json
  --multiple-results MULTIPLE_RESULTS
                        path to the results csv
  --perf-csv PERF_CSV
  --model-name MODEL_NAME
  --common-info COMMON_INFO
```

### Report command - Convert CSV to HTML

Convert CSV to HTML report of models

```
(venv) test-node:~/MAD$ madengine report to-html --help
usage: madengine report to-html [-h] [--csv-file-path CSV_FILE_PATH]

Convert CSV to HTML report of models.

optional arguments:
  -h, --help            show this help message and exit
  --csv-file-path CSV_FILE_PATH
```

### Report command - Convert CSV to Email

Convert CSV to Email report of models

```
(venv) test-node:~/MAD$ madengine report to-email --help
usage: madengine report to-email [-h] [--csv-file-path CSV_FILE_PATH]

Convert CSV to Email of models.

optional arguments:
  -h, --help            show this help message and exit
  --csv-file-path CSV_FILE_PATH
                        Path to the directory containing the CSV files.
```

## Database

Commands for database, such as create and update table of DB.

```
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

### Database - Create Table
```
(venv) test-node:~/MAD$ madengine database create-table --help
usage: madengine database create-table [-h] [-v]

Create table in DB.

optional arguments:
  -h, --help     show this help message and exit
  -v, --verbose  verbose output
```

### Database - Update Table
```
(venv) test-node:~/MAD$ madengine database update-table --help
usage: madengine database update-table [-h] [--csv-file-path CSV_FILE_PATH] [--model-json-path MODEL_JSON_PATH]

Update table in DB.

optional arguments:
  -h, --help            show this help message and exit
  --csv-file-path CSV_FILE_PATH
                        Path to the csv file
  --model-json-path MODEL_JSON_PATH
                        Path to the model json file
```

### Database - Upload MongoDB

```
(venv) test-node:~/MAD$ madengine database upload-mongodb --help
usage: madengine database upload-mongodb [-h] [--type TYPE] [--file-path FILE_PATH] [--name NAME]

Update table in DB.

optional arguments:
  -h, --help            show this help message and exit
  --type TYPE           type of document to upload: job or run
  --file-path FILE_PATH
                        total path to directory where perf_entry.csv, *env.csv, and *.log are stored
  --name NAME           name of model to upload
```

## Tools in madengine

There are some tools distributed with madengine together. They work with madengine CLI to profile GPU and get trace of ROCm libraries.

### Tools - GPU Info Profile

Profile GPU usage of running LLMs and Deep Learning models.

```
(venv) test-node:~/MAD$ madengine run --tags pyt_huggingface_bert --additional-context "{'guest_os': 'UBUNTU','tools': [{'name':'rocprof'}]}"
```

### Tools - Trace Libraries of ROCm

Trace library usage of running LLMs and Deep Learning models. A demo of running model with tracing rocBLAS.

```
(venv) test-node:~/MAD$ madengine run --tags pyt_huggingface_bert --additional-context "{'guest_os': 'UBUNTU','tools': [{'name':'rocblas_trace'}]}"
```

## Environment Variables

Madengine also exposes environment variables to allow for model location setting or data loading at DLM/MAD runtime.

| Field                       | Description                                                                       |
|-----------------------------| ----------------------------------------------------------------------------------|
| MODEL_DIR                   | the location of models dir                                                        |
| PUBLIC_GITHUB_ROCM_KEY           | username and token of GitHub                                                      |
| MAD_AWS_S3                  | the username and password of AWS S3                                               |
| NAS_NODES                   | the list of credentials of NAS Nodes                                              |

Examples for running models using environment variables.
```bash
# Apply AWS S3
MAD_AWS_S3='{"USERNAME":"username","PASSWORD":"password"}' madengine run --tags dummy_data_aws --live-output

# Apply custom NAS
NAS_NODES=[{"HOST":"hostname","PORT":"22","USERNAME":"username","PASSWORD":"password"}] madengine run --tags dummy_data_austin_nas --live-output
```

## Unit Test
Run pytest to validate unit tests of MAD engine.

```
pytest -v -s
```

"""Test the context module.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import sys
import csv
# third-party modules
import pytest
# project modules
from .fixtures.utils import BASE_DIR, MODEL_DIR
from .fixtures.utils import global_data
from .fixtures.utils import clean_test_temp_files
from .fixtures.utils import get_gpu_nodeid_map
from .fixtures.utils import get_num_gpus
from .fixtures.utils import get_num_cpus


class TestContexts:

    @pytest.mark.parametrize('clean_test_temp_files', [['perf.csv', 'perf.html']], indirect=True)
    def test_dockerfile_picked_on_detected_context_0(self, global_data, clean_test_temp_files):
        """ 
        picks dockerfile based on detected context and only those
        """
        global_data['console'].sh("cd " + BASE_DIR + "; " + "MODEL_DIR=" + MODEL_DIR + " " + "python3 src/madengine/mad.py run --tags dummy_ctxtest ") 

        success = False
        with open(os.path.join(BASE_DIR, 'perf.csv'), 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row['model'] == 'dummy_ctxtest':
                    if row['status'] == 'SUCCESS' and row['performance'] == '0': 
                        success = True
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if not success:
            pytest.fail("model did not pick correct context.")

    @pytest.mark.parametrize('clean_test_temp_files', [['perf.csv', 'perf.html', 'ctx_test']], indirect=True)
    def test_dockerfile_picked_on_detected_context_1(self, global_data, clean_test_temp_files):
        """ 
        picks dockerfile based on detected context and only those
        """
        with open(os.path.join(BASE_DIR, 'ctx_test'), 'w') as ctx_test_file:
            print("1", file=ctx_test_file)

        global_data['console'].sh("cd " + BASE_DIR + "; " + "MODEL_DIR=" + MODEL_DIR + " " + "python3 src/madengine/mad.py run --tags dummy_ctxtest ") 

        success = False
        with open(os.path.join(BASE_DIR, 'perf.csv'), 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row['model'] == 'dummy_ctxtest':
                    if row['status'] == 'SUCCESS' and row['performance'] == '1': 
                        success = True
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if not success:
            pytest.fail("model did not pick correct context.")

    @pytest.mark.parametrize('clean_test_temp_files', [['perf.csv', 'perf.html', 'ctx_test']], indirect=True)
    def test_all_dockerfiles_matching_context_executed(self, global_data, clean_test_temp_files):
        """
        All dockerfiles matching context is executed
        """
        with open(os.path.join(BASE_DIR, 'ctx_test'), 'w') as ctx_test_file:
            print("2", file=ctx_test_file)

        global_data['console'].sh("cd " + BASE_DIR + "; " + "MODEL_DIR=" + MODEL_DIR + " " + "python3 src/madengine/mad.py run --tags dummy_ctxtest ") 

        foundDockerfiles = []
        with open(os.path.join(BASE_DIR, 'perf.csv'), 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row['model'] == 'dummy_ctxtest':
                    if row['status'] == 'SUCCESS' and row['performance'] == '2': 
                        foundDockerfiles.append(row['docker_file'].replace(f'{MODEL_DIR}/', '')) 
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if not ("docker/dummy_ctxtest.ctx2a.ubuntu.amd.Dockerfile" in foundDockerfiles and 
           "docker/dummy_ctxtest.ctx2b.ubuntu.amd.Dockerfile" in foundDockerfiles ):
            pytest.fail("All dockerfiles matching context is not executed. Executed dockerfiles are " + ' '.join(foundDockerfiles))

    def test_dockerfile_executed_if_contexts_keys_are_not_common(self):
        """
        Dockerfile is executed even if all context keys are not common but common keys match 
        """
        # already tested in test_dockerfile_picked_on_detected_context_0
        pass

    @pytest.mark.parametrize('clean_test_temp_files', [['perf.csv', 'perf.html']], indirect=True)
    def test_can_override_context_with_additionalContext_commandline(self, global_data, clean_test_temp_files):
        """
        Context can be overridden through additional-context command-line argument 
        """
        global_data['console'].sh("cd " + BASE_DIR + "; " + "MODEL_DIR=" + MODEL_DIR + " " + "python3 src/madengine/mad.py run --tags dummy_ctxtest --additional-context \"{'ctx_test': '1'}\" ") 

        success = False
        with open(os.path.join(BASE_DIR, 'perf.csv'), 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row['model'] == 'dummy_ctxtest':
                    if row['status'] == 'SUCCESS' and row['performance'] == '1': 
                        success = True
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if not success:
            pytest.fail("model did not pick correct context.")

    @pytest.mark.parametrize('clean_test_temp_files', [['perf.csv', 'perf.html', 'ctx.json']], indirect=True)
    def test_can_override_context_with_additionalContextFile_commandline(self, global_data, clean_test_temp_files):
        """
        Context can be overridden through additional-context-file 
        """
        with open(os.path.join(BASE_DIR, 'ctx.json'), 'w') as ctx_json_file:
            print("{ \"ctx_test\": \"1\" }", file=ctx_json_file)

        global_data['console'].sh("cd " + BASE_DIR + "; " + "MODEL_DIR=" + MODEL_DIR + " " + "python3 src/madengine/mad.py run --tags dummy_ctxtest --additional-context-file ctx.json ") 

        success = False
        with open(os.path.join(BASE_DIR, 'perf.csv'), 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row['model'] == 'dummy_ctxtest':
                    if row['status'] == 'SUCCESS' and row['performance'] == '1': 
                        success = True
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if not success:
            pytest.fail("model did not pick correct context.")

    @pytest.mark.parametrize('clean_test_temp_files', [['perf.csv', 'perf.html', 'ctx.json']], indirect=True)
    def test_additionalContext_commandline_overrides_additionalContextFile(self, global_data, clean_test_temp_files):
        """
        additional-context command-line argument has priority over additional-context-file
        """
        with open(os.path.join(BASE_DIR, 'ctx.json'), 'w') as ctx_json_file:
            print("{ \"ctx_test\": \"2\" }", file=ctx_json_file)

        global_data['console'].sh("cd " + BASE_DIR + "; " + "MODEL_DIR=" + MODEL_DIR + " " + "python3 src/madengine/mad.py run --tags dummy_ctxtest --additional-context-file ctx.json --additional-context \"{'ctx_test': '1'}\" ") 

        success = False
        with open(os.path.join(BASE_DIR, 'perf.csv'), 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row['model'] == 'dummy_ctxtest':
                    if row['status'] == 'SUCCESS' and row['performance'] == '1': 
                        success = True
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if not success:
            pytest.fail("model did not pick correct context.")

    @pytest.mark.parametrize('clean_test_temp_files', [['perf.csv', 'perf.html']], indirect=True)
    def test_base_docker_override(self, global_data, clean_test_temp_files):
        """
        BASE_DOCKER overrides base docker
        """
        global_data['console'].sh("cd " + BASE_DIR + "; " + "MODEL_DIR=" + MODEL_DIR + " " + "python3 src/madengine/mad.py run --tags dummy_ctxtest --additional-context \"{'docker_build_arg':{'BASE_DOCKER':'rocm/tensorflow' }}\" ") 

        foundBaseDocker = []
        with open(os.path.join(BASE_DIR, 'perf.csv'), 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row['model'] == 'dummy_ctxtest':
                    if row['status'] == 'SUCCESS' and row['performance'] == '0': 
                        foundBaseDocker.append(row['base_docker']) 
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if not "rocm/tensorflow" in foundBaseDocker:
            pytest.fail("BASE_DOCKER does not override base docker. Expected: rocm/tensorflow Found:" + foundBaseDocker)

    @pytest.mark.parametrize('clean_test_temp_files', [['perf.csv', 'perf.html']], indirect=True)
    def test_docker_image_override(self, global_data, clean_test_temp_files):
        """
        Using user-provided image passed in with MAD_CONTAINER_IMAGE
        """
        global_data['console'].sh("cd " + BASE_DIR + "; " + "MODEL_DIR=" + MODEL_DIR + " " + "python3 src/madengine/mad.py run --tags dummy_ctxtest --additional-context \"{'docker_env_vars':{'ctxtest':'1'},'MAD_CONTAINER_IMAGE':'rocm/tensorflow:latest' }\" ")

        foundLocalImage = None
        with open(os.path.join(BASE_DIR, 'perf.csv'), 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row['model'] == 'dummy_ctxtest':
                    if row['status'] == 'SUCCESS' and row['performance'] == '1':
                        foundLocalImage = row['docker_image']
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if not "rocm/tensorflow:latest" in foundLocalImage:
            pytest.fail("MAD_CONTAINER_IMAGE does not override docker image. Expected: rocm/tensorflow:latest Found:" + foundLocalImage)

    @pytest.mark.parametrize('clean_test_temp_files', [['perf.csv', 'perf.html']], indirect=True)
    def test_docker_env_vars_override(self, global_data, clean_test_temp_files):
        """
        docker_env_vars pass environment variables into docker container 
        """
        global_data['console'].sh("cd " + BASE_DIR + "; " + "MODEL_DIR=" + MODEL_DIR + " " + "python3 src/madengine/mad.py run --tags dummy_ctxtest --additional-context \"{'docker_env_vars':{'ctxtest':'1'} }\" ") 

        success = False
        with open(os.path.join(BASE_DIR, 'perf.csv'), 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row['model'] == 'dummy_ctxtest':
                    if row['status'] == 'SUCCESS' and row['performance'] == '1': 
                        success = True
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if not success:
            pytest.fail("docker_env_vars did not pass environment variables into docker container.")

    @pytest.mark.parametrize('clean_test_temp_files', [['perf.csv', 'perf.html']], indirect=True)
    def test_docker_mounts_mount_host_paths_in_docker_container(self, global_data, clean_test_temp_files):
        """
        docker_mounts mount host paths inside docker containers 
        """
        global_data['console'].sh("cd " + BASE_DIR + "; " + "MODEL_DIR=" + MODEL_DIR + " " + "python3 src/madengine/mad.py run --tags dummy_mountpath --additional-context \"{'docker_env_vars':{'MAD_DATAHOME':'/data'}, 'docker_mounts':{'/data':'/tmp'} }\" ") 

        success = False
        with open(os.path.join(BASE_DIR, 'perf.csv'), 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row['model'] == 'dummy_mountpath':
                    if row['status'] == 'SUCCESS': 
                        success = True
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if not success:
            pytest.fail("docker_mounts did not mount host paths inside docker container.")

    @pytest.mark.skipif(get_num_gpus() < 8, reason="test requires atleast 8 gpus")
    @pytest.mark.parametrize('clean_test_temp_files', [['perf.csv', 'perf.html','results_dummy_gpubind.csv']], indirect=True)
    def test_docker_gpus(self, global_data, clean_test_temp_files):
        """
        docker_gpus binds gpus to docker containers
        """

        global_data['console'].sh("cd " + BASE_DIR + "; " + "MODEL_DIR=" + MODEL_DIR + " " + "python3 src/madengine/mad.py run --tags dummy_gpubind --additional-context \"{'docker_gpus':'0,2-4,5-5,7'}\" ")

        gpu_nodeid_map = get_gpu_nodeid_map()
        with open(os.path.join(BASE_DIR, 'perf.csv'), 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            gpu_node_ids = []
            for row in csv_reader:
                if 'dummy_gpubind' in row['model']:
                    if row['status'] == 'SUCCESS':
                        gpu_node_ids.append(int(row['performance']))
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if sorted(list(map(gpu_nodeid_map.get,gpu_node_ids)))!=[0,2,3,4,5,7]:
            pytest.fail("docker_gpus did not bind expected gpus in docker container.")

    @pytest.mark.skipif(get_num_cpus() < 64, reason="test requires atleast 64 cpus")
    @pytest.mark.parametrize('clean_test_temp_files', [['perf.csv', 'perf.html','results_dummy_cpubind.csv']], indirect=True)
    def test_docker_cpus(self, global_data, clean_test_temp_files):
        """
        docker_cpus binds cpus to docker containers
        """

        global_data['console'].sh("cd " + BASE_DIR + "; " + "MODEL_DIR=" + MODEL_DIR + " " + "python3 src/madengine/mad.py run --tags dummy_cpubind --additional-context \"{'docker_cpus':'14-18,32,44-44,62'}\" ")

        success = False
        with open(os.path.join(BASE_DIR, 'perf.csv'), 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if 'dummy_cpubind' in row['model']:
                    if row['status'] == 'SUCCESS' and row['performance']=="14-18|32|44|62":
                        success = True
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if not success:
            pytest.fail("docker_cpus did not bind expected cpus in docker container.")

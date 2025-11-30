"""Model template for dummy3 model.

This model is used to test the dynamic model discovery feature of MADEngine.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
from madengine.utils.discover_models import CustomModel

Model3Data = CustomModel(
    name="model3",
    dockerfile="docker/dummy",
    scripts="run.sh",
    n_gpus="-1",
    owner="mad.support@amd.com",
    training_precision="",
    tags=["dummies", "dummy_test_group_3"],
    args="",
    multiple_results="",
)

class Dummy3CustomModel(CustomModel):
    def update_model(self):
        self.dockerfile="docker/dummy"
        self.scripts="run.sh"
        self.n_gpus = "-1"
        self.owner = "mad.support@amd.com"
        self.training_precision = ""
        self.args = ""
        self.multiple_results = ""

Model4Data = Dummy3CustomModel(
    name="model4",
    tags = ["dummies", "dummy_test_group_3"]
)

def list_models():
    return [Model3Data, Model4Data]
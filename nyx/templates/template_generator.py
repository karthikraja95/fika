import os
import subprocess
from pathlib import Path


from jinja2 import Environment, PackageLoader

from nyx.templates.util import (
    _create_dir,
    _create_project_dir,
    _get_model_type_kwarg,
)

class TemplateGenerator(object):

    # Prepare environment and source data
    env = Environment(
        loader=PackageLoader("aethos", "templates"),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    project_dir = _create_dir()
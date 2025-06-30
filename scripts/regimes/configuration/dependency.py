from dataclasses import dataclass
from typing import Optional


@dataclass
class Dependency:
    exp_name: str
    run_name: str
    job_id: Optional[int] = None
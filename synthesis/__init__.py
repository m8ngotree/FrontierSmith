"""FrontierSmith open-ended problem synthesis pipeline.

A faithful reproduction of Algorithm 1 from the FrontierSmith paper: closed-ended
competitive-programming problems are mutated into open-ended ones, filtered by a
two-stage idea-divergence funnel, and equipped with agent-built test cases and
verifiers before being emitted in the FrontierCS problem format.
"""

from .config import PipelineConfig

__all__ = ["PipelineConfig"]

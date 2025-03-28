import pkgutil
import importlib

from xpu_graph.passes.patterns.pattern import Pattern, AutoMatchPattern, PatternGroup
from xpu_graph.config import XpuGraphConfig, Target
from xpu_graph.utils import logger


def get_all_patterns(config: XpuGraphConfig):
    patterns = {
        PatternGroup.GROUP0: [],
        PatternGroup.GROUP1: [],
        PatternGroup.GROUP2: [],
    }

    for _, module_name, _ in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{module_name}")

        structure_preplacements = []
        if config.target == Target.mlu:
            from ..targets.mlu.structure_replacements import get_structure_replacements
            structure_preplacements = get_structure_replacements()

        for name in dir(module):
            pat = getattr(module, name)
            if (
                isinstance(pat, type)
                and issubclass(pat, Pattern)
                and pat not in (Pattern, AutoMatchPattern)
                and pat._opt_level <= config.opt_level
            ):
                if pat.__name__ in structure_preplacements:
                    for stage in pat._stages:
                        patterns[pat._pattern_group].append(pat(structure_preplacements[pat.__name__], stage))

    for group, group_patterns in patterns.items():
        logger.debug(f"xpu_graph enable builtin structure {group} patterns: {group_patterns}")

    return patterns

def _wrap_level(gen_mapper):
    def wrapped_level(value):
        res = gen_mapper.map(value)
        if res is None:
            raise ValueError('Generalization step provided has no coverage for this value')
        return res
    return wrapped_level


class GenRule:
    def __init__(self, levels):
        self.levels = levels
        id_fn = lambda x: x
        none_fn = lambda x: None
        custom_levels = [_wrap_level(gen_mapper) for gen_mapper in levels]
        self._levels = [id_fn] + custom_levels + [none_fn]
        self.max_level = len(self._levels) - 1

    def _verify_level(self, level):
        if not isinstance(level, int) or level < 0:
            raise ValueError('Level must be a positive integer')
        if level > self.max_level:
            raise ValueError('Not enough generalization steps available to this rule')
    
    def apply(self, value, level):
        self._verify_level(level)
        return self._levels[level](value)

    def level(self, lvl):
        self._verify_level(lvl)
        return self._levels[lvl]

    def levels_count(self):
        return len(self.levels)

    def get_levels(self):
        return self.levels


class GenMapper:
    def __init__(self, mappings, item_name='', gen_level=None):
        self.mappings = mappings
        self.item_name = item_name
        self.gen_level = gen_level

    def __str__(self):
        return 'generalization of ' + str(self.item_name) + ' with level ' + str(self.gen_level)

    def get_item_name(self):
        return self.item_name

    def get_gen_level(self):
        return self.gen_level

    def get_mappings(self):
        return self.mappings

    def map(self, value):
        return self.mappings[str(value)] if str(value) in self.mappings else None


__all__ = ['GenRule', 'GenMapper']

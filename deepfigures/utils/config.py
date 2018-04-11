"""Classes for serializing data to json."""

import typing

import traitlets


JsonData = typing.Union[list, dict, str, int, float]


class JsonSerializable(traitlets.HasTraits):
    def to_dict(self) -> dict:
        """Recursively convert objects to dicts to allow json serialization."""
        return {
            JsonSerializable.serialize(k): JsonSerializable.serialize(v)
            for (k, v) in self._trait_values.items()
        }

    @staticmethod
    def serialize(obj: typing.Union['JsonSerializable', JsonData]):
        if isinstance(obj, JsonSerializable):
            return obj.to_dict()
        elif isinstance(obj, list):
            return [JsonSerializable.serialize(v) for v in obj]
        elif isinstance(obj, dict):
            res_dict = dict()
            for (key, value) in obj.items():
                assert type(key) == str
                res_dict[key] = JsonSerializable.serialize(value)
            return res_dict
        else:
            return obj

    @classmethod
    def from_dict(cls, json_data: dict):
        assert (type(json_data) == dict)
        args = {}
        for (k, v) in cls.class_traits().items():
            args[k] = JsonSerializable.deserialize(v, json_data[k])
        return cls(**args)

    @staticmethod
    def deserialize(target_trait: traitlets.TraitType, json_data: JsonData):
        """
        N.B. Using this function on complex objects is not advised; prefer to use an explicit serialization scheme.
        """
        # Note: calling importlib.reload on this file breaks issubclass (http://stackoverflow.com/a/11461574/6174778)
        if isinstance(target_trait, traitlets.Instance
                     ) and issubclass(target_trait.klass, JsonSerializable):
            return target_trait.klass.from_dict(json_data)
        elif isinstance(target_trait, traitlets.List):
            assert isinstance(json_data, list)
            return [
                JsonSerializable.deserialize(target_trait._trait, element) for element in json_data
            ]
        elif isinstance(target_trait, traitlets.Dict):
            # Assume all dictionary keys are strings
            assert isinstance(json_data, dict)
            res_dict = dict()
            for (key, value) in json_data.items():
                assert type(key) == str
                res_dict[key] = JsonSerializable.deserialize(target_trait._trait, value)
            return res_dict
        else:
            return json_data

    def __repr__(self):
        traits_list = ['%s=%s' % (k, repr(v)) for (k, v) in self._trait_values.items()]
        return type(self).__name__ + '(' + ', '.join(traits_list) + ')'

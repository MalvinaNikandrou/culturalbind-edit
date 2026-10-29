from abc import ABC, abstractmethod


class Registry:
    _registry = {"task_name_mapping": {}, "method_name_mapping": {}, "dataset_name_mapping": {}}

    @classmethod
    def register(cls, category, name, implementation):
        cls._registry[category][name] = implementation

    @classmethod
    def get(cls, category, name):
        if name not in cls._registry[category]:
            raise ValueError(f"{name} is not a registered {category}.")
        return cls._registry[category][name]

    @classmethod
    def register_task(cls, name, implementation):
        cls.register("task_name_mapping", name, implementation)

    @classmethod
    def get_task(cls, name):
        return cls.get("task_name_mapping", name)

    @classmethod
    def register_method(cls, name, implementation):
        cls.register("method_name_mapping", name, implementation)

    @classmethod
    def get_method(cls, name):
        return cls.get("method_name_mapping", name)

    @classmethod
    def register_dataset(cls, name, implementation):
        cls.register("dataset_name_mapping", name, implementation)

    @classmethod
    def get_dataset(cls, name):
        return cls.get("dataset_name_mapping", name)

    @classmethod
    def register_implementation(cls, category, name):
        def decorator(implementation):
            cls._registry[category][name] = implementation
            return implementation

        return decorator


# Base classes for tasks, methods, models, and datasets
class Task(ABC):
    @abstractmethod
    def run(self):
        pass


class KnowledgeEditingMethod(ABC):
    @abstractmethod
    def edit(self, model, data):
        pass


class Model(ABC):
    @abstractmethod
    def load(self):
        pass


class Dataset(ABC):
    @abstractmethod
    def load_data(self):
        pass


registry = Registry()

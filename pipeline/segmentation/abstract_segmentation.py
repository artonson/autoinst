from abc import ABC, abstractmethod

class AbstractSegmentation(ABC):
    @abstractmethod
    def segment_instances(self, index):
        pass

from abc import ABCMeta, abstractmethod


class UseCase(metaclass=ABCMeta):
    @abstractmethod
    def start(self) -> None:
        pass

import six

from abc import ABCMeta,abstractmethod

@six.add_metaclass(ABCMeta)
class GuessGame_Six(object):

    @abstractmethod
    def message(self, msg):
        pass

    @abstractmethod
    def guess(self):
        pass   
    
    
class ConsoleGame_Six(GuessGame_Six):
    def __init__(self):
        self.welcome = "wwww"
    
    
    def guess(self):
        return int(input(self.prompt))
import logging
from PyQt5 import QtCore

class NoExcFormatter(logging.Formatter):
    def format(self, record):
        record.exc_text = ''
        return super(NoExcFormatter, self).format(record)

    def formatException(self, record):
        return ''


class DictHandler(logging.Handler):
    def __init__(self, log_list):
        super().__init__()
        self.llist = log_list
        self.tfm = NoExcFormatter('%(asctime)s')
        self.lfm = NoExcFormatter('%(levelname)s')
        self.nfm = NoExcFormatter('%(name)s')
        self.mfm = logging.Formatter('%(message)s')
   
    def emit(self, record):
        msg = self.mfm.format(record)
        time = self.tfm.format(record)
        level = self.lfm.format(record)
        name =  self.nfm.format(record)
        
        dic = {'time' : time,
               'level' : level,
               'name' : name,
               'message' : msg}
        
        self.llist.append(dic)

        print(len(self.llist))
        

    

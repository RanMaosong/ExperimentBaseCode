from collections import defaultdict

class Recoder:
    def __init__(self):
        self.metrics = defaultdict(list)

    def record(self, name, value):
        self.metrics[name].append(value)

    def summary(self):
        kvs = {}
        for key in self.metrics.keys():
            kvs[key] = sum(self.metrics[key]) / len(self.metrics[key])
            del self.metrics[key][:]
            self.metrics[key] = []
        return kvs
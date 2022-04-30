import csv
class WeightLogger():
    def __init__(self):
        self.weights = []

    def update_weights(self, weight, tick):
        #print("weights = ", weight)
        list_weight = list(weight)
        list_weight.insert(0, tick)
        list_str_weight = [str(w) for w in list_weight]
        self.weights.append(list_str_weight)
        #print("weights in list  = ", list_weight)

    def end(self, index):
        file = "weights_{}.csv".format(index)
        print("weights = ", self.weights)
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.weights)
        self.weights =[]

# Data source: https://www.kaggle.com/alamson/safebooru
# The data was scraped via Safebooru's online API, then converted from XML to CSV.
# There are 2,736,037 rows of the metadata.
# The data contains images uploaded to safebooru.org in the time range of 2010-01-29 through 2019-06-07.
# The data consists of [id, created_at, rating, score, sample_url, sample_width, sample_height, preview_url, tags]

# encoding = 'utf-8'
import csv
import time
import random
import collections as clt
import numpy as np

time_begin = time.time()

# Get the similarity of two numbers
def Similarity(target, prediction):
    def Loss(target, prediction):
        return abs(target - prediction) ** 2

    loss = Loss(target, prediction)
    if loss > 1:
        return 1/loss
    else:
        return 1

# Loader of CSV file
class CsvLoader():
    # Preprocessing, get the data set list, [[id, score, [tags]], ...]
    def __init__(self, file_path):
        with open(file_path, encoding = 'utf-8', mode = 'r') as file:
            data = csv.reader(file)
            self.dataset = list(data)
            del(self.dataset[0])
            for line in self.dataset:
                del(line[1])                    # delete created_at
                del(line[1])                    # delete rating
                del(line[2])                    # delete sample_url
                del(line[2])                    # delete sample_width
                del(line[2])                    # delete sample_height
                del(line[2])                    # delete preview_url
                line[1] = float(line[1])        # turn the type of score from "str" to "float"
                line[2] = line[2].split(' ')    # split tags

    # Partition processing, get test set list and training set list
    def Divde(self, scale):
        random.shuffle(self.dataset)
        index = len(self.dataset) // scale
        test_list = self.dataset[0:index]
        train_list = self.dataset[index:]
        return test_list, train_list

    # Tags' extraction, get the number of tags and the dictionary of all tags
    # Corresponding content default is its occurrence number
    def TagsExtract(self):
        tags_dict =clt.OrderedDict()
        for line in self.dataset:
            for tag in line[2]:
                if tag not in tags_dict:
                    tags_dict[tag] = 1
                else:
                    tags_dict[tag] += 1
        return tags_dict
    
    # Scores' extraction, get a dictionary{"id": score}
    def ScoresExtract(self):
        scores_dict = dict()
        for line in self.dataset:
            scores_dict[line[0]] = line[1]
        return scores_dict

# Prediction based on KNN
class KNN():
    def __init__(self, train_list, test_list, scores_dict):
        self.train_list = train_list
        self.test_list = test_list
        self.scores_dict = scores_dict

    # For each test data, compute its number of same tags with every train data as the distance
    # The bigger the distance is, the higher their similarity is
    def Distance(self):
        tmp_dict = dict()
        self.dist_dict = dict()
        for line_x in self.test_list:
            for line_y in self.train_list:
                tmp_dict[line_y[0]] = 0
                for tag in line_x[2]:
                    if tag in line_y[2]: # the same tag
                        tmp_dict[line_y[0]] += 1
            self.dist_dict[line_x[0]] = dict(sorted(tmp_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse = True))
    
    # Predict the score of every test data according to it's K most similar train data
    def Predict(self, K):
        prediction_dict = dict()
        for id_x in self.dist_dict:
            cnt = 0
            prediction_dict[id_x] = 0
            for id_y in self.dist_dict[id_x]:
                prediction_dict[id_x] += self.scores_dict[id_y]
                cnt += 1
                if cnt == K: # already get K nearest data
                    break
            prediction_dict[id_x] /= K
        return prediction_dict

    # From K_begin to K_end, test to figure out the best K
    def Test(self, output_file, scores_dict, K_begin, K_end):
        self.Distance()
        with open(output_file, encoding = 'utf-8', mode = 'w') as file:
            max_K = 0
            max_S = 0
            file.write("K\tSimilarity\n")
            for K in range(K_begin, K_end):
                nums = list()
                prediction_KNN = self.Predict(K)
                for id in prediction_KNN:
                    nums.append(Similarity(prediction_KNN[id], scores_dict[id]))
                S = np.mean(nums)
                file.write(str(K) + "\t" + str(S) + "\n")
                if S > max_S:
                    max_K = K
                    max_S = S
            file.write("\nMAX:\n"+str(max_K)+"\t"+str(max_S))

# Prediction based on MLR
class MLR():
    def __init__(self, train_list, test_list, scores_dict, tags_dict):
        self.train_list = train_list
        self.test_list = test_list
        self.scores_dict = scores_dict
        self.tags_dict = tags_dict

    # Construct linear equations and solve them. Y = coeff * X
    def Train(self):
        cnt = 0
        tmp = list()
        Y = list()
        for line in self.train_list:
            Y.append(line[1])
            for tag in self.tags_dict:
                if tag in line[2]:
                    tmp.append(self.tags_dict[tag])
                else:
                    tmp.append(0)
            if cnt == 0:
                X = np.array(tmp)
                cnt += 1
            else:
                X = np.row_stack((X, tmp))
            tmp.clear()
        Y = np.array(Y)
        self.coeff = np.linalg.lstsq(X, Y, rcond = None)[0]

    # Predict the scores according to the coefficients
    def Predict(self):
        cnt = 0
        tmp = list()
        for line in self.test_list:
            for tag in self.tags_dict:
                if tag in line[2]:
                    tmp.append(self.tags_dict[tag])
                else:
                    tmp.append(0)
            if cnt == 0:
                X = np.array(tmp)
                cnt += 1
            else:
                X = np.row_stack((X, tmp))
            tmp.clear()
        cnt = 0
        pre = np.matmul(X, self.coeff)
        prediction_dict = dict()
        for line in self.test_list:
            prediction_dict[line[0]] = pre[cnt]
            cnt += 1
        return prediction_dict
    
    # Test for the similarity
    def Test(self, output_file):
        self.Train()
        prediction_MLR = self.Predict()
        nums = list()
        for id in prediction_MLR:
            nums.append(Similarity(prediction_MLR[id], scores_dict[id]))
        S = np.mean(nums)
        with open(output_file, encoding = 'utf-8', mode = 'w') as file:
            file.write(str(S))
        

scale = 10
input_file = "./Score Prediction/data/data_4w.csv"

# Load data
dataloader = CsvLoader(input_file)
test_list, train_list = dataloader.Divde(scale)
tags_dict = dataloader.TagsExtract()
scores_dict = dataloader.ScoresExtract()

time_load = time.time()

# Run KNN
K_begin = 1
K_end = 51
output_file = "./Score Prediction/result/knn/knn.txt"
Eg_KNN = KNN(train_list, test_list, scores_dict)
Eg_KNN.Test(output_file, scores_dict, K_begin, K_end)

time_knn = time.time()

# Run MLR
output_file = "./Score Prediction/result/mlr/mlr.txt"
Eg_MLR = MLR(train_list, test_list, scores_dict, tags_dict)
Eg_MLR.Test(output_file)

time_mlr = time.time()

# Print time cost
time_file = "./Score Prediction/result/time/time.txt"
with open(time_file, encoding = 'utf-8', mode = 'w') as file:
    file.write("time cost for Loader:\n" + str(time_load - time_begin) + "s\n")
    file.write("time cost for KNN:\n" + str(time_knn - time_load) + "s\n")
    file.write("time cost for MLR:\n" + str(time_mlr - time_knn) + "s\n")
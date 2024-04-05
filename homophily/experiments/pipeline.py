from enum import Enum, auto
from util import *
import random, math
import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Pipeline:
    """
    ideas:
        train model
            - obvious 
        evaluate function
            - would take a function on the three datasets
            - would take a list of constants (if any) OR DATA (if embeddings), somehow take all the parameters that are needed
            - would take list of DATASETS + MODEL types, would only run on these (run on all if null)
            - would RETURN the data in an array of tuples (edges, accuracy, percentage)
            - still output as usual
        save data
            - would take data (map of constants or something + data returned, data can be a tuple)
            - would take name (could combine name and date to save)
            - would take location to store
            - would take expirement description in string form
            - store pandas file!
            - print the location it was stored
        get accuracy of model:
            - given type

        somehow include way to get embeddings from wyatt
    """
    def __init__(self, models):
        if (Model.GCN in models):
            self.gcn = True
            self.initGCN(Datasets.CORA)
        if (Model.SAGE in models):
            self.sage = True
        if (Model.SAINT in models):
            self.saint = True

    def getDataset(self, code):
        if code == Datasets.CORA:
            return self.cora_dataset

    def getModel(self, model):
        if model == Model.GCN:
            return self.cora_gcn
    
    def initGCN(self, data):
        if data == Datasets.CORA:
            self.cora_dataset = Dataset(root='/tmp/Cora', name='Cora', device=device)
            data, in_feats, h_feats, num_classes = self.cora_dataset.get_data()
            
            self.cora_gcn = get_model(in_feats, h_feats, num_classes, "cora")
            self.cora_gcn_ground_truth = get_ground_truth(self.cora_gcn, data, testMask=False)
            print(self.cora_gcn_ground_truth)    

    def evalFunctionOnModel(self, dSet, dModel, func, vals):
        model = self.getModel(dModel)
        for i in vals:
            data = self.getDataset(dSet).get_data()[0]
            modified_graph = data
        
            init_edges = len(modified_graph.edge_index[1])
            
            G, x, y, train_mask, test_mask = convert_to_networkx(modified_graph)
    
            new_G = G.copy()

            func(G, x, y, i, new_G, train_mask, test_mask)
            
            G = new_G.copy()

            modified_graph = convert_to_pyg(G, x, y, train_mask, test_mask)
            final_edges = len(modified_graph.edge_index[1])
            
            output_accuracy_change(self.cora_gcn_ground_truth, test_model(model, modified_graph, testMask=False)) 
            number_added_edges(init_edges, final_edges, is_undirected=True)

class Model(Enum):
    GCN = auto()
    SAGE = auto()
    SAINT = auto()

class Datasets(Enum):
    CORA = auto()
    CITESEER = auto()
    PUBMED = auto()
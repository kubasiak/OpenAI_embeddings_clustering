import openai
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def colorprint(txt,opt="222",end='\n'): 
    #print(f'\033[{opt}m',txt,'\033[0m',end=end)
    print(u"\u001b[38;5;"+opt+'m'+txt+u"\u001b[0m",end=end)

def initialize(engine='text-ada-001'):

    openai.api_type = "azure"
    openai.api_base = os.getenv('OPENAI_API_BASE')
    openai.api_version = "2022-12-01"
    openai.api_key = os.getenv("OPENAI_API_KEY")

    print("openai.api_type: "+openai.api_type)
    print("openai.api_base: "+ openai.api_base)
    print("openai.api_version: "+openai.api_version)
    print("openai.api_key: "+'***')

def project_2D(list_of_arrays_to_project):
    result=[]
    for a in list_of_arrays_to_project:
        tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
        matrix = np.vstack(array_to_project)
        vis_dims2 = tsne.fit_transform(matrix)

        x = [x for x, y in vis_dims2]
        y = [y for x, y in vis_dims2]
        result.append([x,y])
    return(result)


def string2float(l):
    l = [float(x) for x in l]
    return l
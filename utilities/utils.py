import openai
import os
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


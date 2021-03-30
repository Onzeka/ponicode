from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

import pickle
import esprima

import re
import numpy as np
from sklearn.preprocessing import LabelEncoder

from gensim.models import Word2Vec
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from scipy.special import softmax

import sys
modelw2v = Word2Vec.load("./word2vec.model")
modeltfidf = TfidfModel.load("./tfidf.model")
dict_context = Dictionary.load("./tfidfdict.dict")
le = LabelEncoder()
le.classes_ = np.load("classes.npy",allow_pickle=True)

with open('xgb.model','rb') as f:
    model_xgb= pickle.load(f)


@app.route('/', methods=['GET','POST'])
def main():
    return render_template('index.html')

@app.route('/call', methods=['POST'])
def call():
    search_str = request.values.get('search')
    pred = predict(str(search_str))
    print(pred , sys.stderr)
    return jsonify(
        {
            'response': ["coucou"]
        }
    )


if __name__ == "__main__":
    app.run(host="localhost", debug=True)
    
    
def predict(js_code):
    features = get_features(js_code, modeltfidf, modelw2v, dict_context)
    pred = model_xgb.predict(features)
    pred = list(le.inverse_transform(pred))
    return " ".join(pred)
def vectorize_arg(arg, modelw2v):
    norm = 0
    vect = np.zeros(100)
    for a in arg:
        if a in modelw2v.wv:
            vect += modelw2v.wv[a]
            norm += 1
    if norm != 0:
        vect /= norm
    else :
        vect = modelw2v.wv['<param>']
    return vect

def vectorize_context(context,modeltfidf,modelw2v,dict_context):
    tfidf = modeltfidf[dict_context.doc2bow(context)]
    freqs = []
    vects = []
    for id_,freq in tfidf:
        word = dict_context[id_]
        if word in modelw2v.wv:
            vects += [modelw2v.wv[word]]
            freqs += [freq]
    if vects == []:
        vect = modelw2v.wv['<param>']
    else :
        vect = np.stack(vects)
        freqs = softmax(np.array(freqs))
        vect = freqs@vect
    
    return vect

def get_features(js_code,modeltfidf,modelw2v,dict_context):
    args_contexts = extract_args_contexts(js_code)
    features = []
    for arg,context in zip(args_contexts[0],args_contexts[1]):
        features += [np.concatenate((vectorize_arg(arg,modelw2v),vectorize_context(context,modeltfidf,modelw2v,dict_context)))]
    return np.stack(features)

def extract_args_contexts(program: list)->list:
    #récupération de l'ast grace à asprima
    ast = esprima.parseScript(program,loc=True,tokens=True)
    params = ast.body[0].params
    tokens = ast.tokens
    
    # conversion en camel case des paramètres
    args = [re.sub('[A-Z]+',camel,p.name) for p in params]
    
    tokens_values = []
    for t in tokens:
        if t.loc.start.line >= ast.body[0].body.loc.start.line+1:
            # token spécifiques pour string, numeric , regex ce qui permet de réduire le nombre de token en gardant 
            #l'information
            if t.type == 'String':
                tokens_values += ["<s>"]
            elif t.type == 'Numeric':
                tokens_values += ["0"]
            elif t.type == "RegularExpression":
                tokens_values += ["<regex>"]
            else:
                # conversion en camel case des différents token 
                tokens_values += [re.sub('[A-Z]+',camel,t.value)]
    #extraction du body           
    body_raw = [s.split() for s in (" ".join(tokens_values)).split(";")]
    #extraction du context pour chaque argument
    contexts_raw = [[list(filter((arg).__ne__, cs)) for cs in body_raw if arg in cs] for arg in args]
    
    # extraction de l'information syntaxique des token + ajout d'un token de remplacement pour etre en mesure de 
    #vectoriser noms de variables inconnus dans le futur 
    body = []
    for i,cs in enumerate(body_raw):
        body+=[[]]
        for name in cs:
            if name in args:
                body[i] += ['<param>']
            body[i] += break_name(name)
            
    contexts= []
    for i,ctxt in enumerate(contexts_raw):
        contexts += [[]]
        for j, cs in enumerate(ctxt):
            for name in cs:
                contexts[i] += break_name(name) 
    
    args = [break_name(arg) for arg in args]
    
    return np.array([args,contexts])
#on utilise le camel case pour obtenir plus d'information sur les variables 
def break_name(string):
    regex_result = [s.lower() for s in re.findall("[a-zA-Z][a-z]*",string)]
    if string not in ["<s>","0","<regex>"] and regex_result != []:
        return regex_result
    else:
        return [string]
    
def camel(match):
    return match.group(0).title()
from flask import Flask, jsonify, request

import torch
import json

app = Flask(__name__)


model = torch.jit.load('./Models/model_scripted.pt')
lookup_files = ['users_to_i', 'comedy_to_i', 'i_to_users', 'i_to_comedy']
users_to_i, comedy_to_i, i_to_users, i_to_comedy = None, None, None, None
lookup = [users_to_i, comedy_to_i, i_to_users, i_to_comedy]

for i, name in enumerate(lookup_files):
    with open(f'./Lookup/{name}.json', 'r') as json_file:
        lookup[i] = json.load(json_file)

users_to_i, comedy_to_i, i_to_users, i_to_comedy = lookup


def predict_rating(user,top_n):
    user=users_to_i[user]
    comedies=comedy_to_i.keys()
    recomendations=[]
    for x in comedies:
        comedy=comedy_to_i[x]
        input=torch.Tensor([user,comedy]).view(1,-1)
        output=(model(input)).detach().numpy()
        recomendations.append({x:float("{:.4g}".format(output[0][0]*100))})

    recomendations = sorted(recomendations, key=lambda x: list(x.values())[0],reverse=True)
    return {'recomendations':recomendations[:top_n] }


@app.route('/', methods=['GET'])
def get_status():
    return jsonify({'status': 'ready'})

@app.route('/inference', methods=['POST'])
def get_recomendations():
    user=request.json['user_id']
    top_n=request.json['number']
    recomendations=predict_rating(user,top_n)
    return jsonify(recomendations)

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')





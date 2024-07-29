from flask import Flask, request
import json
import os

from agents.DQNAgent import DQNAgent

class Server:
    models = {};
    def __init__(self):
        # model config, t3 
        for file in os.listdir("./models"):
            if file.endswith(".json"):
                #this is modelinfo
                with open(os.path.join("./models/", file)) as f:
                    info = json.loads(f.read())
                    agent = DQNAgent(info["observation_space"], info["action_space"], info["hidden_size"])
                    print(file,file.strip("json"),os.path.join("./models/", file.strip("json")+"pth"))
                    agent.load_model(os.path.join("./models/", file.strip("json")+"pth"))
                    self.models[info["model_id"]]= {"info": info, "agent": agent};
        print("loaded models",self.models)
        self.app = Flask(__name__)
        self.app.route("/", methods=["POST", "GET"])(self.state_handler)
        self.app.run(host='0.0.0.0', port = 5001)

    def state_handler(self):
        if request.headers["x-bhoos-auth"] != os.environ.get("BHOOS_AI_SERVER_AUTH"):
            return json.dumps({"status_failure": "bad auth"});
        data = request.json
        if data != None:
            model = self.models[data["model_id"]];
            if model == None:
                return json.dumps({"status_failure": "model not available"});
            _state = data["state"]
            state: list[int] = _state["state"];
            valid_actions: list[int] = _state["validActions"]
            action_mask = [0 for i in range(self.models[data["model_id"]]["info"]["action_space"])]
            for valid_action in valid_actions:
                action_mask[valid_action] = 1
            obs = {
                'obs': state,
                'action_mask': action_mask
            }
            action = self.models[data["model_id"]]["agent"].get_action(obs)
            return json.dumps({"action": action})
        else: return json.dumps({"status_failure": "bad request"});

server = Server();

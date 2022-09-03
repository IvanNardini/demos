# Deploy a Recommendation System as Hosted Interactive Web Service on AWS

This project represents just one of the stages of my journey about operazionalizing analytics. 
The goal is deploying a Recommendation System as Hosted Interactive Web Service on AWS EC2. 
To achive that, I put together what I learned from IBM’s AI Workflow: Enterprise Model Deployment certification on Coursera and other seminaries I attented from Databricks. 


## The Scenario

Because I'm not interested about modelling, I goggled a bit and I find this interesting article “How to Build a Recommendation System for Purchase Data (Step-by-Step)”. In this article, the scenario is 

*A grocery chain releases a new mobile app allowing its customers to place orders before they even have to walk into the store. So when customer taps on the “order” page, the recommend top 10 items just pop up ready to be added to the cart.*

Based on that, I start thinking about a possible machine learning system and I assume

1. **Data scientist runs and tracks many model experiments based on staging purchase data. Then, they register the Champion model in the Mlflow registry**
2. **Machine Learning engineer hands over the model and builds the scoring function**
3. **Together with the DevOps engineer, they deploy the model as Hosted Dash Interactive Web Service on AWS**


Below the **high-level architecture of the project**: 

<p align="center">
<img src="https://github.com/IvanNardini/modelops-aws-web-endpoint-hosted/raw/master/architecture.png">
</p>

And, at the end, this is the Dash Interactive Web Service I get

<p align="center">
<img src="https://github.com/IvanNardini/modelops-aws-web-endpoint-hosted/raw/master/modelops_app.gif" >
</p>

## Usage 

Depending on which enviroment you want to run (Experimentation, Development and Test environment or Staging, Preproduction and Production environment) you can run: 

```
docker-compose -f docker-compose.dev.yml build
docker-compose -f docker-compose.dev.yml up
```

```
docker-compose -f docker-compose.dev.yml -f docker-compose.stage.yml build 
docker-compose -f docker-compose.dev.yml -f docker-compose.stage.yml build 
```
Once you prototype the dashboard, you can deploy it on AWS EC2 Instance with 

```
# update local packages
sudo apt-get update -y
# install dependencies
sudo apt-get install -y python3-pip python3-dev python3-venv
# create the python enviroment
python3 -m venv pyenv
# activate a virtual environment¶
source ./pyenv/bin/activate
#install packages
pip install -r ./src/score_interactive_endpoint/requirements.txt
#run application
chmod +x ./run.py
python ./run.py
```

## Contributing

Test it. And please provide me feedback for improvements. Pull requests are welcome as well.

And feel free to reach me at [Ivan Nardini](ivan.nardini@sas.com )

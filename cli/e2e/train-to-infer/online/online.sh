#Create compute if it does not exists already
az ml compute create -n cpu-cluster --type amlcompute

#Submit the training job , get the run id
run_id=$(az ml job create -f jobs/single-step/scikit-learn/iris/job.yml --query name -o tsv)

echo "Using run id : $run_id"

#wait until job finishes
status=$(az ml job show -n $run_id --query status -o tsv)
running=("Queued" "Starting" "Preparing" "Running" "Finalizing")
while [[ ${running[*]} =~ $status ]]
do
  sleep 8 
  status=$(az ml job show -n $run_id --query status -o tsv)
  echo $status
done

#Download job logs including model, into the current path , a path create with the run id
az ml job download -n $run_id

#Copy the model into the model path for deployment to pick it up.

cp $run_id/model/model.pkl ./model/model.pkl

#Generate endpoint name randomly
export ENDPOINT_NAME=endpt-`echo $RANDOM`

echo "Using endpoint name : $ENDPOINT_NAME"

#Create an online endpoint
az ml online-endpoint create --name $ENDPOINT_NAME

# create a deployment, 
az ml online-deployment create --name blue --endpoint $ENDPOINT_NAME -f blue-deployment.yml --all-traffic


az ml online-endpoint invoke --name $ENDPOINT_NAME --request-file endpoints/online/model-1/sample-request.json

az ml online-endpoint delete --name $ENDPOINT_NAME -y
#!/bin/bash -e

sep="######################"

ext=$(az extension show -n ml --query version -o tsv)
group=$(az config get defaults.group  --query value -o tsv)
workspace=$(az config get defaults.workspace  --query value -o tsv)


echo "$sep Starting this scenario with default resource group: $group , workspace: $workspace, and  ml extension version: $ext"

#Comment out and create compute name "cpu-cluster" for inference batch job if it does not exists already
#az ml compute create -n cpu-cluster --type amlcompute --max-instances 5

#Comment out and create compute name "gpu-cluster" for training job if it does not exists already
#az ml compute create -n gpu-cluster --type amlcompute


echo "$sep Creating the training job "
run_id=$(az ml job create -f jobs/training.yml --web --query name -o tsv)


echo "$sep Checking status of the job with run id $run_id, press ^Ctrl+C to come out "
status=$(az ml job show -n $run_id --query status -o tsv)


running=("Queued" "Starting" "Preparing" "Running" "Finalizing")

while [[ ${running[*]} =~ $status ]]
do
  sleep 8
  status=$(az ml job show -n $run_id --query status -o tsv)
  echo $status
done


echo "$sep Download job logs including model, into the current path with  $run_id path name "
az ml job download -n $run_id


echo "$sep Copying model to model folder path "
cp $run_id/model/model.pth ./model/model.pth

#Generate endpoint name randomly
export ENDPOINT_NAME=endpt-`echo $RANDOM`

echo "Create batch endpoint"
az ml batch-endpoint create -n $ENDPOINT_NAME

echo "Create deployment"
az ml batch-deployment create -f blue-deployment.yml -n red -e $ENDPOINT_NAME --set-default


echo "Create a dataset from the local directory with scoring images"
az ml dataset create -n batch-inf-data --local_path sample_request/

echo "$sep Invoking endpoint "

az ml batch-endpoint invoke --name $ENDPOINT_NAME --input-dataset azureml:batch-inf-data:1  --output-path folder:azureml://datastores/workspaceblobstore/paths/myoutpu
t --set output_file_name=pred.csv --mini-batch-size 20 --instance-count 5 --query name -o tsv

echo "$sep Deleting endpoint "
az ml batch-endpoint delete --name $ENDPOINT_NAME -y
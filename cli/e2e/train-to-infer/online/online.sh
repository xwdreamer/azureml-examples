

#Comment out and create compute name "cpu-cluster" if it does not exists already
#az ml compute create -n cpu-cluster --type amlcompute

echo "Creainge the training job "
run_id=$(az ml job create -f jobs/scikit-learn/iris/job.yml --query name -o tsv)


echo " Checking status of the job with run id $run_id, press ^Ctrl+C to come out "
status=$(az ml job show -n $run_id --query status -o tsv)


running=("Queued" "Starting" "Preparing" "Running" "Finalizing")
while [[ ${running[*]} =~ $status ]]
do
  sleep 8
  status=$(az ml job show -n $run_id --query status -o tsv)
  echo $status
done


echo " Download job logs including model, into the current path with  $run_id path name "
az ml job download -n $run_id

#Copy the model into the model path for deployment to pick it up.

echo " Copying model to model folder path "

cp $run_id/model/model.pkl ./model/model.pkl

#Generate endpoint name randomly
export ENDPOINT_NAME=endpt-`echo $RANDOM`


echo " Creating endpoint $ENDPOINT_NAME "
az ml online-endpoint create --name $ENDPOINT_NAME

echo " Creating deployment "
az ml online-deployment create --name blue --endpoint $ENDPOINT_NAME -f blue-deployment.yml --all-traffic

echo " Invoking endpoint "
az ml online-endpoint invoke --name $ENDPOINT_NAME --request-file sample-request.json

echo " Deleting endpoint "
az ml online-endpoint delete --name $ENDPOINT_NAME -y
source ../.env/bin/activate

python3 resnet.py > resnet.log 2>&1 &

../../notify.sh

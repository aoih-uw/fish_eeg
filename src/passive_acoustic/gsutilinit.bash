echo "Changing to deps dir"
# Get the directory where this script itself lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR

cd ../..
cd deps/


echo "Downloading gsutil for importing google cloud files"
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
echo "Unpacking linux installation"
tar -xf google-cloud-cli-linux-x86_64.tar.gz
echo "Add gsutil CLI to path"
./google-cloud-sdk/install.sh
echo "Initialize gsutil"
./google-cloud-sdk/bin/gcloud init

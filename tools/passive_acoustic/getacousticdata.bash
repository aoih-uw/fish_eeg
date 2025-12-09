echo "Make sure to run gsutilinit.bash first"
sleep 2
echo "Pulling Acoustic data..."

# Get the directory where this script itself lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# cd to dir where script resides 
cd $SCRIPT_DIR

# cd back to fisheeg home
cd ../..

# make a data dir if needed
[ -d "data" ] || mkdir -p "data"

cd data
echo "Moving to separate data folder"
mkdir -p PAcoustic
cd PAcoustic

gsutil -m cp -r gs://noaa-passive-bioacoustic/dclde/2026/dclde_2026_killer_whales/orcasound/* .


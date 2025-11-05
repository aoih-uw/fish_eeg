echo "Make sure to run gsutilinit.bash first"
sleep 3
echo "Pulling Acoustic data..."
cd data
echo "Moving to separate data folder"
mkdir -p PAcoustic
cd PAcoustic

gsutil -m cp -r gs://noaa-passive-bioacoustic/dclde/2026/dclde_2026_killer_whales/orcasound/* .


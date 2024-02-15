#!/bin/bash

kTs=(0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)
dl="dataset1_"
gsd=".gsd"
out="out/"

for kT in "${kTs[@]}"
do
  python3 simulate_cubes_hoomd.py --N 100 --L 26.2 --output "$dl$kT" --ts 500000 --dump 100 --gpu 1 --kT $kT --dt 0.005
  python3 getpairs.py -i "$out$dl$kT$gsd" --cutoff 6.3
  rm "$out$dl$kT$gsd"
done

cfs=`ls ./out/raw_data/*figs.txt`
for cf in $cfs
do
  tf="${cf/configs/"torqueforce"}"
  python3 process_geometry.py --configs $cf --torfors $tf
done

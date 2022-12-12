#!/bin/bash
cd /home/kusch/Documents/project/patient_analyse/paper/presentation_2/figure/
for i in */*;
#for i in *.pdf;
  do
    echo $i;
    gimp -i -b '(batch-autocrop-file "'$i'")' -b '(gimp-quit 0)';
    done;



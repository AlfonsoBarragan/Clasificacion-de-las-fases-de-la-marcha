#!/bin/bash
echo "Killing process..."
name=$(ps -e | grep hcidump)

for i in $name; do
    if [[ $i =~ ^[0-9]+$ ]]; then
        echo "Process ${i} killed"
        kill $i 
    fi
done

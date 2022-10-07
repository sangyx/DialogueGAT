for year in 2015 2016 2017; do
    for target in 3 7 15; do
        echo -e "\nDialogueGAT [year: ${year} target: ${target}]"
        python -W ignore -u train.py --use_gpu --v_past --year $year --target $target
    done
done
echo "Done!"

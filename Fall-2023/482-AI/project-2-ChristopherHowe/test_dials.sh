pythonFile="car.py"
bestAlpha=0
bestGamma=0
bestResult=0

trials=1
numDifferentVals=5

for ((j=1; j<=numDifferentVals; j++)); do
    alpha=$(echo "scale=1; $j/$numDifferentVals" | bc)
    for ((i=1; i<=numDifferentVals; i++)); do
        gamma=$(echo "scale=1; $i/$numDifferentVals" | bc)
        result=$(python3 $pythonFile --train --alpha $alpha --gamma $gamma 2> /dev/null | grep 'best attempt' | awk '{print $NF}')
        if [ $(echo "$result >= $bestResult" | bc -l) -eq 1 ]; then
            bestResult=$result
            bestAlpha=$alpha
            bestGamma=$gamma
        fi
        echo trial $trials alpha: $alpha gamma: $gamma result: $result
        ((trials++))
    done
done

echo bestResult: $bestResult, bestAlpha: $bestAlpha, bestGamma, $bestGamma


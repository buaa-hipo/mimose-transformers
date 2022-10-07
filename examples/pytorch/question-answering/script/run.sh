set -x

for i in 2 4 6 8
do
    sh ./run_qa_womax_dc.sh $i
    sh ./run_qa_womax_dc_static.sh $i
done
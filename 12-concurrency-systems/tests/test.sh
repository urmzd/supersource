#!/bin/bash

do_diff() {
  # $1 expected
  # $2 is produced
  # $3 Test name

  echo -n "$3 test "
  diff $1 $2 > /dev/null
  if [ $? != 0 ]; then
    echo FAILED
    echo ======
    echo Expected Output:
    echo ++++++++++++++++
    cat $1
    echo =====================
    echo Actual Output:
    echo ++++++++++++++++
    cat $2
    exit 1
  else
    echo PASSED
  fi
}


echo ======================================================
echo ====================== TEST $1 ========================
echo ======================================================
ACTUAL=tests/test.$1.out
ACT_BASIC=tests/test.$1.act_bsc
ACT_RECV=tests/test.$1.act_recv
ACT_ONLY=tests/test.$1.act_only
ACTUAL2=tests/test.$1.out2
ACT2_RECV=tests/test.$1.act2_recv
EXPECT=tests/test.$1.expected
EXP_BASIC=tests/test.$1.exp_bsc
EXP_RECV=tests/test.$1.exp_recv
EXP_ONLY=tests/test.$1.exp_only

CFG=`cat tests/test.$1.cfg`
DELAY=tests/delay.py
INPUT=tests/test.$1.in

if which timeout > /dev/null; then
  TIMEOUT="timeout 10"
  echo "##" Using a timeout of 10 seconds
  echo ======================================================
else
  TIMEOUT=""
fi

cat tests/test.$1.desc
echo ======================================================
cat $INPUT | python3 $DELAY $CFG | $TIMEOUT ./miner/miner > $ACTUAL
EXIT=${PIPESTATUS[2]}
if [ $EXIT == 0 ]; then 
  grep -v ^Thread $ACTUAL > $ACT_RECV
  grep -v ^Thread $EXPECT > $EXP_RECV
  grep -v ^Received $ACT_RECV > $ACT_BASIC
  grep -v ^Received $EXP_RECV > $EXP_BASIC
  do_diff $EXP_BASIC $ACT_BASIC "Basic functionality"

  if grep BASIC tests/test.$1.cfg > /dev/null; then
    grep ^Received $ACTUAL > $ACT_ONLY
    grep ^Received $EXPECT > $EXP_ONLY
    
    do_diff $EXP_ONLY $ACT_ONLY "Receive statement ordering should be the same"
  elif grep RANDOM tests/test.$1.cfg > /dev/null; then
    cat $INPUT | python3 $DELAY $CFG | $TIMEOUT ./miner/miner > $ACTUAL2
    grep -v ^Thread $ACTUAL2 > $ACT2_RECV

    echo Input has been randomly delayed to create random behaviour
    echo Output from multiple runs should be different 
    diff $ACT_RECV $ACT2_RECV > /dev/null
    if [ $? == 0 ]; then
      echo Output from multiple runs is the same: test FAILED
      exit 1
    else
      echo Output from multiple runs is different: test PASSED
    fi
  else
    do_diff $EXP_RECV $ACT_RECV "Receive ordering should be the same"
  fi
elif [ $EXIT == 124 ]; then
  echo TIMEOUT
  exit 1
else 
  echo Abnormal program termination: the program crashed
  echo Exit code $?
  exit 1
fi

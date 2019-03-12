#!/bin/bash
 
mkdir .locks 2>/dev/null
# lock dirs/files
LOCKDIR="$@"
LOCKDIR=${LOCKDIR//[ \/.]/_}
LOCKDIR=".locks/$LOCKDIR"
PIDFILE="${LOCKDIR}/PID"
 
# exit codes and text
ENO_SUCCESS=0; ETXT[0]="ENO_SUCCESS"
ENO_GENERAL=1; ETXT[1]="ENO_GENERAL"
ENO_LOCKFAIL=2; ETXT[2]="ENO_LOCKFAIL"
ENO_RECVSIG=3; ETXT[3]="ENO_RECVSIG"
 
###
### start locking attempt
###
 
trap 'ECODE=$?; echo "[once.sh] Exit: ${ETXT[ECODE]}($ECODE)" >&2' 0
#echo -n "[once.sh] Locking: " >&2
 
if mkdir "${LOCKDIR}" &>/dev/null; then
 
    # lock succeeded, install signal handlers before storing the PID just in case 
    # storing the PID fails
    trap 'ECODE=$?;
          #echo "[once.sh] Removing lock. Exit: ${ETXT[ECODE]}($ECODE)" >&2
          rm -rf "${LOCKDIR}"' 0
    echo "$$" >"${PIDFILE}" 
    # the following handler will exit the script upon receiving these signals
    # the trap on "0" (EXIT) from above will be triggered by this trap's "exit" command!
    trap 'echo "[once.sh] Killed by a signal." >&2
          exit ${ENO_RECVSIG}' 1 2 3 15
    #echo "success, installed signal handlers"
    pname=$1
    shift
    #echo $pname "$@"
    $pname "$@"
    rm $PIDFILE
    rmdir $LOCKDIR
else
 
    # lock failed, check if the other PID is alive
    OTHERPID="$(cat "${PIDFILE}")"
 
    # if cat isn't able to read the file, another instance is probably
    # about to remove the lock -- exit, we're *still* locked
    #  Thanks to Grzegorz Wierzowiecki for pointing out this race condition on
    #  http://wiki.grzegorz.wierzowiecki.pl/code:mutex-in-bash
    if [ $? != 0 ]; then
      echo "[once.sh] run $@ failed, PID ${OTHERPID} is active" >&2
      exit ${ENO_LOCKFAIL}
    fi
 
    if ! kill -0 $OTHERPID &>/dev/null; then
        # lock is stale, remove it and restart
        #echo "[once.sh] removing stale lock of nonexistant PID ${OTHERPID}" >&2
        rm -rf "${LOCKDIR}"
        #echo "[once.sh] restarting myself" >&2
	DIR=$(cd $(dirname "$0"); pwd)
	exec $DIR/`basename $0` "$@"
        exec "$0" "$@"
    else
        # lock is valid and OTHERPID is active - exit, we're locked!
        echo "[once.sh] run $@ failed, PID ${OTHERPID} is active" >&2
        exit ${ENO_LOCKFAIL}
    fi
 
fi
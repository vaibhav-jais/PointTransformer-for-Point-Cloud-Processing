#!/bin/bash -e
if ! grep -q '^172.26.3.89 gitlab.aptiv.today$' /etc/hosts && [ "$EUID" == 0 ]; then
    echo '172.26.3.89 gitlab.aptiv.today' >> /etc/hosts
fi

if [ "$EUID" == 0 ]; then
    PROXY=http://10.214.44.76:3128
else
    PROXY=http://10.233.48.50:8080
fi
export no_proxy=".aptiv.today"
cp /results/cluster_key/private_shared/setup_cluster/putty_private_key.ppk /tmp/ cluster_key
chmod 600 /tmp/cluster_key

export HDF5_USE_FILE_LOCKING='FALSE'

if [ ! -v GIT_SSH_COMMAND ]; then
	export GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no -i /tmp/cluster_key'
fi

if [ "$1" != "" ]; then
    REFERENCE=$1
else
    REFERENCE="dev"
fi

if [ ! -v HOST ]; then
    HOST="ssh://git@gitlab.aptiv.today:2289/mlperception/vaibhav-masterthesis.git"
fi
# TARGET="/tmp/perl_local"
if [ ! -v TARGET ]; then
	TARGET=$(mktemp -d -t vaibhav-masterthesis-XXXXXXXXX)
else
    mkdir -p $TARGET
fi

# git clone repo:
echo "Fetching $REFERENCE from $HOST into $TARGET"

## Add more folder to include into LFS using KOMMA (,). Example: LFS_WHITELIST="End2End/torch,End2End/model")
LFS_WHITELIST="End2End/torch,models" ## Later is the new folder structure

do_full_clone() {
  git clone $HOST $TARGET || return
  cd $TARGET || return
  git checkout $REFERENCE
  return
}
do_shallow_clone() {
	cd $TARGET || return
	rm -rf .* *
	git init || return
	git remote add origin $HOST || return
	git fetch --depth 1 origin $REFERENCE || return
	GIT_LFS_SKIP_SMUDGE=1 git reset --hard FETCH_HEAD|| return
	## Pull only the whitelisted lfs files (this can be a comma, separated list of multiple subdirecotories)
	git lfs pull --include ${LFS_WHITELIST}
	return
}


COUNTER=0
MAX_TRIES=20
if [ "$2" == "-f" ]; then
	until do_full_clone; do
	    COUNTER=$[$COUNTER + 1]
		echo "Trying to get repo, trial $COUNTER out of $MAX_TRIES"
		if [ $COUNTER -ge $MAX_TRIES ]; then echo "Problem cloning repository, exiting script"; exit 2; fi
		sleep 5;
	done;
else
	until do_shallow_clone; do
	    COUNTER=$[$COUNTER + 1]
		echo "Trying to get repo, trial $COUNTER out of $MAX_TRIES"
		if [ $COUNTER -ge $MAX_TRIES ]; then echo "Problem cloning repository, exiting script"; exit 2; fi
		sleep 5;
	done;
fi


if [ "$(whoami)" == "root" ]; then 
    USER=$(tail -n1 /etc/passwd | cut -d: -f1)
fi
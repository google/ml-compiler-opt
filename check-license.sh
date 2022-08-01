# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

exitcode=0

check_license () {
  HEADER=$(head -1 $1)
  if [[ $HEADER == \#!* ]] # Ignore shebang line
  then
    HEADER=$(head -$3 $1 | sed 1d)
  else
    HEADER=$(head -$(($3 - 1)) $fname)
  fi
  if [[ "$HEADER" != "$2" ]]
  then
    echo "$1 does not have license header. Please copy and paste ./license-header.txt"
    exitcode=1
  fi
}

for fname in `find . -name "*.py"`;
do
  check_license $fname "$(cat license-header.txt)" 15
done

for fname in `find . -name "*.sh"`;
do
  check_license $fname "$(tail -13 license-header.txt)" 14
done

exit ${exitcode}


## Overview

In this demo we will look at:

* Building LLVM with the correct settings to allow for model training
* Collecting a training corpus for the regalloc model based on Chromium
* Training the regalloc model on the collected corpus
* Compiling the trained model into LLVM

## Preliminaries

Set up some environment variables according to where you want to clone/build
all of the code:
```bash
export WORKINGD_DIR=~
```

Change the directory to wherever you'd like to put everything.

## Get repositories

### ml-compiler-opt (this repository)

Clone the github repo:

```bash
cd $WORKING_DIR
git clone https://github.com/google/ml-compiler-opt
```

### LLVM

Grabbing LLVM should be as simple as running the below command, but if
something goes awry, make sure to check the
[official documentation](https://llvm.org/docs/GettingStarted.html).

```bash
git clone https://github.com/llvm/llvm-project.git
```

### Chromium

Grabbing Chromium is a bit more involved. The 
[official documentation](https://chromium.googlesource.com/chromium/src/+/main/docs/linux/build_instructions.md)
for Linux based systems (the only platform currently supported by MLGO) is
available at that link. However, cloning the code should be as simple as
downloading depot_tools and adding them to your `$PATH`:

```bash
git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
export PATH="$PATH:$WORKING_DIR/depot_tools"
```

Creating a folder for Chromium and cloning it using the `fetch` utility:
```bash
mkdir chromium
cd chromium
fetch --nohooks --no-history chromium
```

**Note:** Running `fetch` with the `--no-history` flag makes your local
checkout significantly smaller and speeds up the underlying git clone by a
significant amount. However, if you actually need the history for any
reason (eg to revert to a previous commit or to work on the Chromium side
with MLGO stuff), make sure to omit this flag to do a full checkout.

The fetch command will take at least a couple minutes on a fast internet
connection and much longer on slower ones.

Next, we need to modify the `.gclient` file that `fetch` creates in the
directory that you run it in to make sure that the Chromium PGO profiles
get checked out:

```bash
sed -i 's/"custom_vars": {},/"custom_vars": { "checkout_pgo_profiles" : True },/' .gclient
```

This `sed` command will set the necessary variable correctly. After this,
you can move into the `src` directory that `fetch` created that contains
the actual Chromium codebase. Now, we need to apply the bitcode embedding
patch contained in this repository.

```bash
cd src
git apply $WORKING_DIR/ml-compiler-opt/experimental/chromium-bitcode-embedding.patch
git apply $WORKING_DIR/ml-compiler-opt/experimental/chromium-thinlto-corpus-extraction.patch
```

This will make a `clang_embed_bitcode` flag and a `lld_emit_index` flag
available in the gn configuration.

Now that this is all in place, you need to run the Chromium hooks in order to
get the development environment ready for a full compilation:

```bash
gclient runhooks
```

## Install Dependencies

If you're working in a Debian based docker container, it will most likely
not come by default with `sudo`. It isn't stricly necessary to install,
but it makes it easier to copypaste the installation commands below and it
also enables the use of the Chromium dependency auto-installation script:

```bash
apt-get install sudo
```

First, install some base dependencies that will be needed when building
LLVM:

```bash
sudo apt-get install cmake ninja-build lld
```

Now, install the Chromium dependencies using the auto-installation script:
```bash
$WORKING_DIR/chromium/src/build/install-build-deps.sh
```

**Note:** These installation commands are all designed to be run on Debian
based distros. However, adapating to other distros with alternative package
management systems should not be too difficult. The packages for the first
command should be very similarly named and the
[official Chromium documentation](https://chromium.googlesource.com/chromium/src/+/main/docs/linux/build_instructions.md)
has info on dependency installation for their build process on other common
distros.

Also make sure that you install the Python dependencies for the
ml-compiler-opt repository:

```bash
cd $WORKING_DIR
pip3 install -r ml-compiler-opt/requirements.txt
```

If you plan on doing development work on this checkout of ml-compiler-opt,
use the `/ml-compiler-opt/requirements-ci.txt` requirements file to install
some additional development dependencies.

## Building Chromium

**WARNING:** Currently, Chromium only builds with protobuf 4.x.x while Tensorflow
requires protobuf 3.x.x. In order to make sure that Chromium compiles correctly
you can either use a virtual environment and install protobuf 4.x.x there (currently
just the latest version), or you can install protobuf 4.x.x over the currently
installed version and then undo it later after the compile is complete. This tutorial
assumes no usage of virtual environments. Install a compatible version of protobuf:

```bash
pip3 install protobuf==4.21.7
```

To build Chromium, make sure you are in the `/chromium/src` directory and then
run the following command to open a CLI text editor that will allow you to
configure build settings:

```bash
cd $WORKING_DIR/chromium/src
gn args ./out/Release
```

Then, you need to specify the configuration options to use when compiling Chromium.
This will depend upon the type of training corpus that you want to extract. If you
want to extract a non-thinLTO corpus, you can use the configuration listed below:

```
is_official_build=true
use_thin_lto=false
is_cfi=false
clang_embed_bitcode=true
is_debug=false
symbol_level=0
enable_nacl=false
```

But if you want to extract a thinLTO corpus, you need to use the following config:

```
is_official_build=true
lld_emit_index=true
is_debug=false
symbol_level=0
enable_nacl=false
```

Immedaitely after closing the editor, `gn` will generate all of the files
necessary so that `ninja` can execute all the necessary compilation steps.
However, to extract a corpus for ML training, we also need a database of
compilation commands. This can be obtained by running the following command:

```
gn gen ./out/Release --export-compile-commands
```

Then you can build Chromium with the `autoninja` utility:

```bash
autoninja -C ./out/Release
```

A full Chromium compile will take at least an hour on pretty well specced
hardware (ie 96 thread work station) and much longer on lower specced
hardware.

TODO(boomanaiden154): Investigate the source of this assertion error.

**Note:** If the build fails in the last couple steps tripping an assertion in
the linker when compiling a non-thinLTO corpus, you can safely ignore this. 
Preparing a corpus for ML training in the non-thinLTO case only requires the
object files that get fed to the linker

**WARNING:** make sure to reinstall a version of protobuf compatible with the
current tf-nightly release used by ml-compiler-opt if you changed versions earlier
to get the Chromium compile working. Reinstalling using the ml-compiler-opt lockfile
should work:

```bash
pip3 install -r $WORKING_DIR/ml-compiler-opt/requirements.txt
```

## Building LLVM

To build LLVM to train ML models, we first need to build TFLite and some
dependencies so that we can embed it within LLVM to load and execute models
on the fly during reinforcement learning. There is a script within this
repository that clones and builds everything automatically and prepares
a CMake cache file that can be passed to CMake during the LLVM build
configuration. Running the script looks like this:

```bash
cd $WORKING_DIR
mkdir tflite
cd tflite
$WORKING_DIR/ml-compiler-opt/buildbot/build_tflite.sh
```

This script should only take a couple minutes to execute as all the libraries
that it pulls and builds are relatively small.

Now, create a new folder to do an LLVM build and configure it using CMake:
```bash
mkdir $WORKING_DIR/llvm-build
cd $WORKING_DIR/llvm-build
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -C $WORKING_DIR/tflite/tflite.cmake \
  $WORKING_DIR/llvm-project/llvm
```

Now you can run the actual build with the following command:
```bash
cmake --build .
```

## ML training

All of the following example commands assume you are working from within
your checkout of the ml-compiler-opt repository:

```bash
cd $WORKING_DIR/ml-compiler-opt
```

To start off training, we need to extract a corpus from the Chromium compile.
The procedure for this will depend upon how you built your corpus, particularly
whether or not you used thinLTO.

### Corpus extraction (non-thinLTO case)

For corpus extraction in the non-thinLTO case, you can simply run the following
command:

```bash
PYTHONPATH=$PYTHONPATH:. python3 compiler_opt/tools/extract_ir.py \
  --cmd_filter="^-O2|-O3" \
  --input=$WORKING_DIR/chromium/src/out/Release/compile_commands.json \
  --input_type=json \
  --llvm_objcopy_path=$WORKING_DIR/llvm-build/bin/llvm-objcopy \
  --output_dir=$WORKING_DIR/corpus
```

This command will extract all the relevant bitcode and compilation flags from
the Chromium compile and put them in the `$WORKING_DIR/corpus` directory. No
further processing on the corpus should be needed.

### Corpus extraction (thinLTO case)

Corpus extraction for the thinLTO case is slightly more involved. Start off by
running the following command to do the initial step in the corpus extraction
process:

```bash
PYTHONPATH=$PYTHONPATH:. python3 compiler_opt/tools/extract_ir.py \
  --cmd_filter="^-O2|-O3" \
  --input=$WORKING_DIR/chromium/src/out/Release/compile_commands.json \
  --input_type=json \
  --llvm_objcopy_path=$WORKING_DIR/llvm-build/bin/llvm-objcopy \
  --output_dir=$WORKING_DIR/corpus \
  --thinlto_build=local \
  --obj_base_dir=$WORKING_DIR/chromium/src/out/Release/obj
```

After this, it is necessary to grab the flags passed to the linker and add
them to the `corpus_description.json` file in the `$WORKING_DIR/corpus` folder.
To find this, it is helpful to look at the actual invocation of the linker. To
see the linker invocation for a target like `chrome`, go to the Chromium build
directory and run the following command:

```bash
cd $WORKING_DIR/chromium/src/out/Release
ninja -t commands chrome
```

The last command should look something like this:

```bash
python3 "../../build/toolchain/gcc_link_wrapper.py" --output="./chrome" -- ../../third_party/llvm-build/Release+Asserts/bin/clang++ -Wl,--version-script=../../build/linux/chrome.map -Werror -fuse-ld=lld -Wl,--fatal-warnings -Wl,--build-id=sha1 -fPIC -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now -Wl,--icf=all -Wl,--color-diagnostics -Wl,-mllvm,-instcombine-lower-dbg-declare=0 -Wl,--save-temps=import -Wl,--thinlto-emit-index-files -flto=thin -Wl,--thinlto-jobs=all -Wl,--thinlto-cache-dir=thinlto-cache -Wl,--thinlto-cache-policy=cache_size=10\%:cache_size_bytes=40g:cache_size_files=100000 -Wl,-mllvm,-import-instr-limit=30 -fwhole-program-vtables -Wl,--undefined-version -m64 -no-canonical-prefixes -Wl,-O2 -Wl,--gc-sections -rdynamic -Wl,-z,defs -Wl,--as-needed -nostdlib++ --sysroot=../../build/linux/debian_bullseye_amd64-sysroot -fsanitize=cfi-vcall -fsanitize=cfi-icall -pie -Wl,--disable-new-dtags -Wl,--lto-O2 -o "./chrome" -Wl,--start-group @"./chrome.rsp"  -Wl,--end-group  -ldl -lpthread -lrt -lgmodule-2.0 -lgobject-2.0 -lgthread-2.0 -lglib-2.0 -lnss3 -lnssutil3 -lsmime3 -lplds4 -lplc4 -lnspr4 -latk-1.0 -latk-bridge-2.0 -lcups -lgio-2.0 -ldrm -ldbus-1 -latspi -lresolv -lm -lX11 -lXcomposite -lXdamage -lXext -lXfixes -lXrender -lXrandr -lXtst -lgbm -lEGL -lexpat -luuid -lxcb -lxkbcommon -lXi -lpci -l:libffi_pic.a -lpangocairo-1.0 -lpango-1.0 -lharfbuzz -lcairo -lasound -lz -lstdc++ -lxshmfence
```

From this, we can take out a couple flags that we need for our `corpus_description.json`. A more
precise description of the flags that are needed is available in the [MLGO ThinLTO documentation](../thinlto.md). Given the linker command above, the flags that we need consist of the following:

```
-fPIC
-mllvm,-instcombine-lower-dbg-declare=0
-mllvm,-import-instr-limit=30
-no-canonical-prefixes
-O2
-nostdlib++
--sysroot=../../build/linux/debian_bullseye_amd64-sysroot
-c
```

Make sure to rewrite the `--sysroot` flag to be an absolute path. Setting it
to the output of the following should work:

```bash
echo $WORKING_DIR/chromium/src/build/linux/debian_bullseye_amd64-sysroot
```

Now, add the commands to the `global_command_override` section in the 
`corpus_description.json` file. Afterwards, the `global_command_override` 
section in the file should look something like the following:

```json
"global_command_override": [
  "-fPIC",
  "-mllvm",
  "-instcombine-lower-dbg-declare=0",
  "-mllvm",
  "-import-instr-limit=30",
  "-no-canonical-prefixes",
  "-O2",
  "-nostdlib++",
  "--sysroot=/path/to/workdir/build/linux/debian_bullseye_amd64-sysroot",
  "-c"
]
```

Now you should have a properly prepared Chromium thinLTO corpus.

### PGO Path Rewriting

**NOTE:** This step is only necessary if you are working on a non-thinLTO
corpus. If you are working on a thinLTO corpus, making changes outlined in the
section below will result in an error when you try and generate a default trace.

It is essential to have PGO data when training the regalloc model. However,
the Chromium build process uses relative paths when referencing PGO profiles.
This needs to be fixed by replacing the default `-fprofile-instrument-use-path`
flag with one that uses an absolute path. First we need to know the profile
being used. The profiles are located in
`$WOKRING_DIR/chromium/src/chrome/build/pgo_profiles` and they will have a
`*.profdata` extension. If you have never resynced your checkout, there should
be just one file. Then to find the absolute path of the file, run the following
command:

```bash
realpath -s $WORKING_DIR/chromium/src/chrome/build/pgo_profiles/*.profdata
```

This should output an absolute path to a `*.profdata` file. Then, open the
gin config for the regalloc problem which should be located at
`compiler_opt/rl/regalloc/gin_configs/common.gin` within the ml-compiler-opt
root. Then, replace the line

```
problem_config.flags_to_replace.replace_flags={}
```

With this:
```
problem_config.flags_to_replace.replace_flags = {
  '-fprofile-instrument-use-path': '<path to profdata from above command>'
}
```

eg:

```
problem_config.flags_to_replace.replace_flags = {
  '-fprofile-instrument-use-path': '/home/aiden/chromium/src/chrome/build/pgo_profiles/chrome-linux-main-1665359392-180123bdd1fedc45c1cbb781e2ad04bd98ab1546.profdata'
}
```

Adding a flag to remove warnings related to PGO profdata hash mismatches
can also be helpful to not clutter up your output. These can be added in
by adjusting this line:

```
problem_config.flags_to_add.add_flags=()
```

to

```
problem_config.flags_to_add.add_flags=('-Wno-backend-plugin',)
```

### Collect the Default Trace and Generate Vocab

Before we run reinforcement training, it is best to train the model using
behavioral cloning on the default heuristic. First off, we need to collect
a trace of the decisions the default heuristic is making:

```bash
PYTHONPATH=$PYTHONPATH:. python3 compiler_opt/tools/generate_default_trace.py \
  --data_path=$WORKING_DIR/corpus \
  --output_path=$WORKING_DIR/default_trace \
  --gin_files=compiler_opt/rl/regalloc/gin_configs/common.gin \
  --gin_bindings=clang_path="'$WORKING_DIR/llvm-build/bin/clang'" \
  --sampling_rate=0.2
```

This will compile 20% of the corpus and save all of the regalloc eviction
problem instances it encounters into the `/default_trace` file.

After we have collected a default trace, an optional step is to regenerate the
vocab that is used to normalize some of the values that get fed to the ML
model:

```bash
rm -rf ./compiler_opt/rl/regalloc/vocab
PYTHONPATH=$PYTHONPATH:. python3 compiler_opt/tools/sparse_bucket_generator.py \
  --input=$WORKING_DIR/default_trace \
  --output_dir=./compiler_opt/rl/regalloc/vocab \
  --gin_files=compiler_opt/rl/regalloc/gin_configs/common.gin
```

This isn't completely necessary as there are already default values stored
within the repository in the `./compiler_opt/rl/regalloc/vocab` folder,
but it definitely doesn't hurt to regenerate them. If adding/modifying
features, it is necessary to regenerate the vocab.

Now that the vocab is present (or has been regenerated) and we have a default
trace, we can start to train the model using behavioral cloning to mimic the
default heuristic:

```bash
PYTHONPATH=$PYTHONPATH:. python3 compiler_opt/rl/train_bc.py \
  --root_dir=$WORKING_DIR/warmstart \
  --data_path=$WORKING_DIR/default_trace \
  --gin_files=compiler_opt/rl/regalloc/gin_configs/behavioral_cloning_nn_agent.gin
```

This script shouldn't take too long to run on decently powerful hardware.
It will output a trained model in the directory specified by the `--rootdir`
flag, in this case `$WORKING_DIR/warmstart`.

## Reinforcement Learning

Now that we have a model that has been warmstarted based on the default
heuristic, we can now proceed with RL learning so that the model can
improve beyond the performance of the default heuristic:

```bash
PYTHONPATH=$PYTHONPATH:. python3 compiler_opt/rl/train_locally.py \
  --root_dir=$WORKING_DIR/output_model \
  --data_path=$WORKING_DIR/corpus \
  --gin_bindings=clang_path="'$WORKING_DIR/llvm-build/bin/clang'" \
  --gin_files=compiler_opt/rl/regalloc/gin_configs/ppo_nn_agent.gin \
  --gin_bindings=train_eval.warmstart_policy_dir=\"$WORKING_DIR/warmstart/saved_policy\"
```

This script will take quite a while to run. Probably most of the day on pretty
powerful hardware (~100+ vCPUs), and potentially many days on less powerful
hardware.

## Evaluting the Policy

If you interested in seeing how the trained policy performs, you can go
through two different avenues. You can run the `generate_default_trace.py`
script to get info on the reward (reduction in number of instructions) over
a specific corpus. However, this still doesn't tell the whole story for the
regalloc case and actual benchmarking is needed. There is also some tooling
available in this repository to run benchmarks in Chromium and the
llvm-test-suite using performance counters to track instructions executed,
loads, and stores, all of which are metrics that show how the model is
performing.

### Evaluating the Model With Reward Metrics

To evaluate a trained policy (for example looking at the output from
RL training in `$WORKING_DIR/output_model/saved_policy`), run the
`generate_default_trace.py` script with some flags to tell it to output
performance data:

```bash
PYTHONPATH=$PYTHONPATH:. python3 compiler_opt/tools/generate_default_trace.py \
  --data_path=$WORKING_DIR/corpus \
  --gin_files=compiler_opt/rl/regalloc/gin_configs/common.gin \
  --gin_bindings=config_registry.get_configuration.implementation=@configs.RegallocEvictionConfig \
  --gin_bindings=clang_path="'$WORKING_DIR/llvm-build/bin/clang'" \
  --output_performance_path=$WORKING_DIR/performance_data.csv \
  --policy_path=$WORKING_DIR/output_model/saved_policy
```

This will collect reward data over the entire corpus. If you want it to run
faster and don't care about collecting data over the whole corpus, you can
set the `--sample_rate` flag to a desired value to only operate over a portion
of the corpus.

### Evaluating the Model With Benchmarking

See the documentation available [here](../benchmarking.md)

## Deploying the New Policy

To compile the model into LLVM using Tensorflow AOT compilation,
create a new folder and run a CMake configuration with the following
commands:

```bash
mkdir $WORKING_DIR/llvm-release-build
cd $WORKING_DIR/llvm-release-build
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORFLOW_AOT_PATH=$(python3 -c "import tensorflow; import os; print(os.path.dirname(tensorflow.__file__))") \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DLLVM_RAEVICT_MODEL_PATH="$WORKING_DIR/output_model/saved_policy" \
  $WORKING_DIR/llvm-project/llvm
```

Then run the actual build:

```bash
cmake --build .
```

Now, you should have a build of clang in `$WORKING_DIR/llvm-release-build/bin` that
you can use to compile projects using the ML regalloc eviction heuristic.
To compile with the ML regalloc eviction heuristic, all you need to do
is make sure to pass the `-mllvm -regalloc-enable-advisor=release` flag
to `clang` whenever you're compiling something.
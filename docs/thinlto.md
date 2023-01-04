# Extracting Corpora for ThinLTO-enabled Projects

These steps use inlining for size as an example.

## Generate the corpus

Pass `-thinlto-emit-index-files -save-temps=import` to lld, this will output `.3.import.bc`
and `.thinlto.bc` files in your source directory.   

## Consolidate the corpus

Run `tools/extract_ir.py` with `--thinlto_build=local` and set `--obj_base_dir`, `--output_dir`
accordingly.

## Modify corpus_description.json

Modification of `corpus_description.json` which is written to `output_dir` is necessary. An error will
be thrown if this is not done. \
The `global_command_override` field in the json needs to be filled with the options to run
**clang** with on each module. These options should be inferred from the lld command that generated the
corpus. Most importantly, it should include "-c", some "-O" flag, relevant mllvm flags, and target/arch
flags. \
\
Here's an example based on Chrome:
```json
{
  "global_command_override": [
    "-fPIC",
    "-mllvm",
    "-instcombine-lower-dbg-declare=0",
    "-mllvm",
    "-import-instr-limit=5",
    "-march=armv7-a",
    "--target=arm-linux-androideabi23",
    "-no-canonical-prefixes",
    "-O2",
    "-nostdlib++",
    "--sysroot=/path/to/linux-x86_64/sysroot",
    "-c"
  ]
}
```

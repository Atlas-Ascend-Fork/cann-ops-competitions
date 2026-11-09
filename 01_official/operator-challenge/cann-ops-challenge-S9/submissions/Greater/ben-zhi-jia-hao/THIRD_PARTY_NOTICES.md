# Third-party build support notices

The build support files in this project are derived from the public
[cann/cann-ops-competitions repository](https://gitcode.com/cann/cann-ops-competitions)
at commit `f390918f2083f477b59831c0808af5bc0af6a949`.

## Source locations

- Root `CMakeLists.txt`, `CMakePresets.json`, and `build.sh`:
  `01_official/operator-challenge/cann-ops-challenge-S8/submissions/Assign/ccate/`.
- The `cmake/` directory and `scripts/help.info`, `scripts/install.sh`, and
  `scripts/upgrade.sh`:
  `01_official/operator-challenge/cann-ops-challenge-S8/submissions/Fills/终不似，少年游/`.

Root build configuration and shell entry points may contain local integration
changes. The original copyright and license notices in the support files are
preserved. This provenance list covers build support files; it does not assign a
new license to the operator source files.

## License notices

- A verbatim copy of the upstream repository license is included as
  `third_party_licenses/cann-ops-competitions-LICENSE.txt`. Its original placeholder
  copyright line is retained as published upstream.
- Huawei copyright notices remain in the CANN support scripts. In particular,
  `cmake/util/ascendc_gen_options.py` and `cmake/util/parse_ini_to_json.py` carry
  Apache License 2.0 notices. The license text is included as
  `third_party_licenses/Apache-2.0.txt`.
- The bundled Makeself source and its notices are retained in
  `cmake/util/makeself/`, including its `COPYING` file. Makeself identifies its own
  license as GNU GPL version 2 or later. Its upstream notice states that archives
  created with Makeself do not inherit the license of the packaging tool.

Keep the applicable copyright notices, license notices, and license texts when
redistributing these build support files.

## Local build integration changes

The local configuration resolves the SDK directory from the environment and
selects native OPP packaging. The shell entry script performs a clean build,
stops on failure, and checks that packaging produced an installation file.
The CMake helper handles an empty compiler launcher, invokes a shell helper
through Bash, and stops when source generation fails.

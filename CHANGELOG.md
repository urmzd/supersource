# Changelog

## 0.8.0 (2026-10-07)

### Features

- **paths**: split the FDE path into six parts composed by Superstar FDE ([3e4b0ea](https://github.com/urmzd/supersource/commit/3e4b0eaf0d00e31816f291ffc15ebb3a268f5c08))
- **practice**: let paths include other paths in ss learn ([57963b1](https://github.com/urmzd/supersource/commit/57963b1d9bca9960313ceb0e83f22a59e36220c9))

### Misc

- **llm-systems**: add serving architecture diagrams in D2 and Mermaid ([2f2a00a](https://github.com/urmzd/supersource/commit/2f2a00a2698b277db3510f830845009c7eada84b))
- **field-engineering**: add the field engineering track ([3261d29](https://github.com/urmzd/supersource/commit/3261d2989e02835ebd8efa95a1877d367ca7951d))
- **book**: build composed path PDFs and fail on stale diagram renders ([9767d83](https://github.com/urmzd/supersource/commit/9767d83e87985ba2cd48761919906ee623e580c5))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.7.0...v0.8.0)


## 0.7.0 (2026-10-07)

### Features

- **paths**: add the FDE inference role path and wire it into the curriculum ([1056748](https://github.com/urmzd/supersource/commit/10567487ad377d22b18a93500eda595ff3860104))
- **practice**: add ss learn to read role paths stage by stage ([f3ca3a0](https://github.com/urmzd/supersource/commit/f3ca3a06baf003bf987296304fe63779c73061ab))
- **practice**: pin the JDK with mise and resolve it inside ss ([be14be3](https://github.com/urmzd/supersource/commit/be14be3358343d9c7835dccd97ad3e0e03e3548f))
- **practice**: complete the C++ track with five more exercises ([4496480](https://github.com/urmzd/supersource/commit/449648076236fba85d04fbf93654fae0a92f3423))
- **practice**: add five C++ build exercises ([723f09d](https://github.com/urmzd/supersource/commit/723f09d7b2c3f57143d48330687b4bd969815e34))
- **practice**: add ten verified C build exercises ([e8467ed](https://github.com/urmzd/supersource/commit/e8467edcb1d176619a06e04c926bd62ec8d9fb7e))

### Bug Fixes

- **practice**: include stdint.h and memory explicitly so gcc builds the references ([8469e86](https://github.com/urmzd/supersource/commit/8469e8659bd1eee0ea2ca4dfe799e96b3d3f3d70))

### Refactoring

- **llm-systems**: rename the serving platforms guide to match its title ([dfcd60d](https://github.com/urmzd/supersource/commit/dfcd60d565018d67941bb69b11cedb04294ca9be))
- **practice**: split build languages into core and optional ([c70086c](https://github.com/urmzd/supersource/commit/c70086c08c05b2a18ac73c1e8c7cef29e1bf745b))
- **practice**: consolidate predict and build into one practice path ([f96c416](https://github.com/urmzd/supersource/commit/f96c416341c5f930db145e6bdfbf48763d5c76b5))

### Misc

- **llm-systems**: apply ruff format to the model loading scripts ([3b1f5b0](https://github.com/urmzd/supersource/commit/3b1f5b005097d49fda2ecb40046f72291ca63a41))
- add CS curriculum and validated source register ([bc9ac9f](https://github.com/urmzd/supersource/commit/bc9ac9fce7610ecff439659d587dd4e141335745))
- **book**: build and release one PDF per role path ([1844ffb](https://github.com/urmzd/supersource/commit/1844ffb20ea2cc75b9c79147d96889fd874197aa))
- **ml**: add training and post-training topic with LoRA, DPO, and memory tooling ([59cf089](https://github.com/urmzd/supersource/commit/59cf0897cc00e01b663b083ae0908f90ffeffe7b))
- **llm-systems**: add serving and load deep dive with capacity calculator and load generator ([a8b001d](https://github.com/urmzd/supersource/commit/a8b001d4a57b63654fe8232de596312bf9adede3))
- **llm-systems**: add model loading deep dive with safetensors and config tooling ([e65a686](https://github.com/urmzd/supersource/commit/e65a686203f6ba070bce35c2a85d9a0d289b660e))
- point measure tests at practice/bin after the merge ([516ec07](https://github.com/urmzd/supersource/commit/516ec070c71429cd2bac70040ea38b38888e1f74))
- **llm-systems**: add serving platforms guide and link it from framework docs ([f801646](https://github.com/urmzd/supersource/commit/f8016467ce1a031617e63443fd07d6375f0e880a))
- **case-studies**: add Agent Evaluation Harness worked build ([a5c2e2a](https://github.com/urmzd/supersource/commit/a5c2e2a91893461361b99d388fe0742715e92f88))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.6.0...v0.7.0)


## 0.6.0 (2026-09-27)

### Features

- **practice**: assert wall time and peak memory per snippet ([ffb1b74](https://github.com/urmzd/supersource/commit/ffb1b74154ec199b05048edd77df17ef0d59933f))
- **practice**: fill the predict harness with 24 verified snippets ([c08fea3](https://github.com/urmzd/supersource/commit/c08fea3522c5ec950703c304e13308fd0ccacbf9))
- **practice**: add predict-then-run harness baseline ([ae50471](https://github.com/urmzd/supersource/commit/ae50471ea036db911c96443befb3e1df418f3c1e))
- **programming-languages**: add Type Systems & Polymorphism track (#14) ([438d929](https://github.com/urmzd/supersource/commit/438d9295c7dad5940a210105c98a1cbfe1f6b7be))

### Bug Fixes

- **ci**: measure child RSS without Python launcher memory ([b1f3a00](https://github.com/urmzd/supersource/commit/b1f3a00ea4d0583e03af12cde526a2dbb1e0e9dc))

### Misc

- **platform**: cross-link routing track and serving platforms guide ([fc00373](https://github.com/urmzd/supersource/commit/fc003732f91740041b643f6018b0eaf38ee5cb5c))
- **platform**: cover System One decision models and specialised models in the pool ([bd330ca](https://github.com/urmzd/supersource/commit/bd330ca8778a7119dca955f028950260df855954))
- **case-studies**: explain what text-to-SQL benchmarks can and cannot tell you ([5ab5473](https://github.com/urmzd/supersource/commit/5ab54731e6694c48e9ea930cede7bf62257947cf))
- **platform**: add Model Routing & Cascades track ([fda5cc7](https://github.com/urmzd/supersource/commit/fda5cc79e47292fd2df954816388100571755b94))
- explain LLM serving platforms and routing trade-offs ([9fdcacb](https://github.com/urmzd/supersource/commit/9fdcacb6c251bb6414628de4a7c418d62e2862b9))
- **case-studies**: add worked-build track consolidating four projects ([781847d](https://github.com/urmzd/supersource/commit/781847d084809ef186e4040ad26a9c7c65c08da6))
- **craftsmanship**: add Lessons from Practice, consolidating the lessons archive ([426640d](https://github.com/urmzd/supersource/commit/426640d1d95bfcfa3b594feaa328cf9e17ce2c75))
- bump GitHub Actions to Node 24 majors and switch app token to client-id ([0af765e](https://github.com/urmzd/supersource/commit/0af765e43b730c9f86012fec10f55a8e9618afcd))
- bump setup-chrome to v2 for Node 24 support ([12673d4](https://github.com/urmzd/supersource/commit/12673d493b4aafbedad041769b91901aa2e49352))
- pin setup-uv to v7 (no floating v8 major tag exists) ([b6c4188](https://github.com/urmzd/supersource/commit/b6c4188f0efddafb870ead127fff56a53fc4a3e6))
- sync uv.lock to 0.5.0 and add it to sr stage_files ([bd61a10](https://github.com/urmzd/supersource/commit/bd61a1073526c0948c93f9af4b58c696b5d3e207))
- bump all GitHub Actions to latest Node 24 majors ([0a92da2](https://github.com/urmzd/supersource/commit/0a92da2d01c2ca327e6acda915d54616a4fd5d7e))
- bump checkout to v5 and setup-uv to v6 for Node 24 support ([b8ca677](https://github.com/urmzd/supersource/commit/b8ca6774728aa66717087f980c2ae0a625f6980f))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.5.0...v0.6.0)


## 0.5.0 (2026-06-14)

### Features

- **data-engineering**: add runnable Streamflow orders pipeline (Spark + dbt + Airflow) ([4e565e9](https://github.com/urmzd/supersource/commit/4e565e99c86b733c83c5cdff1111534c6b209e7e))

### Misc

- **data-engineering**: ruff-format Spark job and sync uv.lock to 0.4.0 ([b54e9cb](https://github.com/urmzd/supersource/commit/b54e9cb97f51baa746a7db7e2dce9cb72bb6b46a))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.4.0...v0.5.0)


## 0.4.0 (2026-06-14)

### Features

- **infrastructure**: split diagramming-and-operations into focused tracks ([abc2225](https://github.com/urmzd/supersource/commit/abc222582df212a63a58dfaa11cc9cdcf8226e55))

### Misc

- **infrastructure**: apply ruff format to demo code (#13) ([cefb6de](https://github.com/urmzd/supersource/commit/cefb6dedff0d6c1ad827408e74e112ab42e825e3))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.3.1...v0.4.0)


## 0.3.1 (2026-06-14)

### Bug Fixes

- **book**: one chapter per topic and single section numbering (#10) ([a6ed4a1](https://github.com/urmzd/supersource/commit/a6ed4a139fab15cadbc29724e571b596da1e9186))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.3.0...v0.3.1)


## 0.3.0 (2026-06-13)

### Features

- **ml**: Neural Architectures & the History of Deep Learning (#8) ([e3627d1](https://github.com/urmzd/supersource/commit/e3627d19e9240b2168f4f1de0881cae6fafbe935))

### Misc

- **ml**: add LLM development lifecycle section to Foundation Models (#9) ([d60bc49](https://github.com/urmzd/supersource/commit/d60bc49b463fdd0d16e8b9364f57bc9f0442e858))
- attach curriculum PDF as a release asset via sr (#7) ([5ee5cc8](https://github.com/urmzd/supersource/commit/5ee5cc85cda4e80948045f8e2033ed7055139582))
- build the whole curriculum into one learnable PDF (#6) ([ca2a1a6](https://github.com/urmzd/supersource/commit/ca2a1a62600211e318abc3d6b11b3bacd3aade4a))
- add Diagramming & Operations track (#5) ([41074eb](https://github.com/urmzd/supersource/commit/41074eb134426d4102361a2e0c53709fd8bd9df4))
- add data engineering track and LLM systems & inference topic (#4) ([d769f70](https://github.com/urmzd/supersource/commit/d769f70eaeb04a1930351851a840ff1c61d9ac06))
- **ci**: bump sr to v8 ([b3ba9d3](https://github.com/urmzd/supersource/commit/b3ba9d383e8ea9f0586cf0796335c9e553ac1fa0))
- **ci**: remove unused force input from release workflow ([90c4dc5](https://github.com/urmzd/supersource/commit/90c4dc57f51e76193bd7048e04bf6a2d22338d48))
- **fix**: add CI badge to README ([4841e57](https://github.com/urmzd/supersource/commit/4841e57f8fa77ead1b6e6172b4cc882634f99000))
- update internal references from superpowers to supersource ([559cba3](https://github.com/urmzd/supersource/commit/559cba3f901c603cc2f9670f8881eb91d1b5165b))
- **community**: add GitHub community-health files ([0892131](https://github.com/urmzd/supersource/commit/08921314378366be496771e9f101ecee9c2670c5))
- **fix**: rewrite README with standard structure ([ee66c07](https://github.com/urmzd/supersource/commit/ee66c07a757ea1d2fca44120a38af40bd03c8257))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.2.1...v0.3.0)


## 0.2.1 (2026-04-16)

### Bug Fixes

- **ci**: migrate sr v4 to v7 for artifact and input support (#3) ([eefd537](https://github.com/urmzd/supersource/commit/eefd537226c35bd8615ea3a8bf65b1853929a6fd))

### Misc

- migrate sr config and action to v4 ([67a66c3](https://github.com/urmzd/supersource/commit/67a66c3c4b794d35623915ae7fad6dc7cbff98fd))
- add linguist overrides to fix language stats ([8b0132f](https://github.com/urmzd/supersource/commit/8b0132fcce12972090b150d1c9fba18f289a60d6))
- update sr action from v2 to v3 ([ce2dfb6](https://github.com/urmzd/supersource/commit/ce2dfb67d32a391a85f6103dc691effa4acb768e))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.2.0...v0.2.1)


## 0.2.0 (2026-03-30)

### Features

- **swiss-table**: scaffold Swiss Table hash table implementation ([8b15450](https://github.com/urmzd/supersource/commit/8b15450a989777cec383784b073dd3bd9936788e))
- **skills**: add study-session skill for interactive learning ([8645990](https://github.com/urmzd/supersource/commit/86459906d20eeb42a61c7742428392966398e2c0))
- **skills**: add study-plan skill for personalized learning ([2c7a645](https://github.com/urmzd/supersource/commit/2c7a64573030e8d33f6e87aaf33f95626aaa30b8))
- **skills**: add practice-impl skill for scaffolding implementations ([a715ff2](https://github.com/urmzd/supersource/commit/a715ff2081526f6e348f1f490ea6e09739d9f07d))
- add mathematics curriculum foundation ([9f2a1e7](https://github.com/urmzd/supersource/commit/9f2a1e76b40a8f2acc0fd6b04ebc62994f6d6a97))
- **patterns**: add existing pattern guides for all topics ([197b976](https://github.com/urmzd/supersource/commit/197b976e2633ce611b7b389de69b9b04102ce5ba))
- **patterns**: add ML & Statistics patterns guide ([1e56195](https://github.com/urmzd/supersource/commit/1e5619559e4c6c043b9412630e7d2f65b42e680c))
- **interviews**: add comprehensive interview guides for 19 top companies ([84884f9](https://github.com/urmzd/supersource/commit/84884f9e7a9054df3cf36ac288c36d18f1f7f4e8))
- **probabilistic-structures**: add bloom filter event validator ([abb499f](https://github.com/urmzd/supersource/commit/abb499fe190e7d69e2afd9099411ed0df00923aa))
- **ml-stats**: add linear regression implementation ([9de1eda](https://github.com/urmzd/supersource/commit/9de1eda76eb3a88a2c49b365e681be258a8685cb))
- **greedy**: implement Huffman coding compression algorithm ([d1d6f94](https://github.com/urmzd/supersource/commit/d1d6f9462866ea847fb050d4592887eac92ecf15))
- **graphs**: add Charles and the Corgi Conundrum solution template ([fdeb949](https://github.com/urmzd/supersource/commit/fdeb9498efef11049fce520662fbf0551316d317))
- **practice**: add TF-IDF vector search implementation ([e4a5ddd](https://github.com/urmzd/supersource/commit/e4a5ddd7cbe0a2b9fcd81999f6568c8dcf6c602a))
- **practice**: add k-means clustering implementation ([03efa32](https://github.com/urmzd/supersource/commit/03efa32381ebb10ea26161d1e4320a238c648f60))
- **main**: add entry point script ([8abb5b3](https://github.com/urmzd/supersource/commit/8abb5b3430cfca0869ab7db8796d4756df636d19))

### Bug Fixes

- auto-fix lint errors and suppress pre-existing violations in ruff config ([44b7fb1](https://github.com/urmzd/supersource/commit/44b7fb128116e71ec00ee51aefd7aa2e97b8162b))
- add ruff and pytest as dev dependencies ([dcc9cbc](https://github.com/urmzd/supersource/commit/dcc9cbcbc17d88f055f49a8dcbb50ebd62efe679))
- remove --group dev from ci.yml — no dev dependency group defined ([64ea772](https://github.com/urmzd/supersource/commit/64ea77222f3ffe02a8d79db6f5f88a10ad22b6e2))

### Documentation

- **software-craftsmanship**: add professional development track documentation ([bb7f674](https://github.com/urmzd/supersource/commit/bb7f674eaff0bb6eb8b4087ecf7189896bafdd59))
- **practice**: add polyglot practice guide for nine languages ([1a02774](https://github.com/urmzd/supersource/commit/1a027745f2d552da0acea698539422dd27e59728))
- **systems**: add comprehensive README for systems and architecture tracks ([f4cb6e2](https://github.com/urmzd/supersource/commit/f4cb6e2a5c0a293bea6dc896dacc426b7ae39e60))
- **ml**: add comprehensive README for all machine learning tracks ([51e3fdf](https://github.com/urmzd/supersource/commit/51e3fdf4088a65d2bfd743264b9a222b25a6af6f))
- **math**: add comprehensive README for all mathematics tracks ([b722838](https://github.com/urmzd/supersource/commit/b7228386dc33f6ad0d2674cb72b6ec228d92db7f))
- **interviews**: add OpenAI preparation resources ([2da8c50](https://github.com/urmzd/supersource/commit/2da8c501ea641134fa206ea17e52be54a5238811))
- **curriculum**: expand with ML/AI, systems, competitive programming, and information theory tracks ([c5588c5](https://github.com/urmzd/supersource/commit/c5588c5115d2b6d20ab3b2253956d8bb284e9d32))
- add structured 21-day study plan ([c513458](https://github.com/urmzd/supersource/commit/c513458504aeee62c7cbaa7934cfc9fefef29497))
- restructure main readme for new curriculum ([115d5cb](https://github.com/urmzd/supersource/commit/115d5cb042cbdda309373e586c03e085c3ce5466))
- add agent skill following agentskills.io spec ([3ac34ca](https://github.com/urmzd/supersource/commit/3ac34ca46aefd294e5536741501fb9e97f3c6033))
- rewrite root README with linked topics table and 21-day study plan ([de08ad2](https://github.com/urmzd/supersource/commit/de08ad2db1f2c9da05195d2f560e35fe345529b8))
- **ml-stats**: add clustering analysis conclusions ([9ae32c7](https://github.com/urmzd/supersource/commit/9ae32c747d8233fc38686b8268e7dc3ac5f7d71b))

### Refactoring

- **algorithms**: move ML practice implementations to canonical location ([0d2232d](https://github.com/urmzd/supersource/commit/0d2232d012f26011b6903985147d3e00884f3992))
- rename project from superpowers to supersource ([2e7380b](https://github.com/urmzd/supersource/commit/2e7380b3b572b657bd420dc51cae31816a08e534))
- move algorithms to dedicated directory ([fafdc4e](https://github.com/urmzd/supersource/commit/fafdc4ebe67b43e2b145397c587ecc0aaa4fc0b4))
- move algorithms to dedicated directory ([462ee88](https://github.com/urmzd/supersource/commit/462ee88c6febc884162ff0e20503fa87417a57fb))
- move pattern guides into topic READMEs ([3bbdcaf](https://github.com/urmzd/supersource/commit/3bbdcaf1cbc015910854372a1c5210848cbd52e9))
- **ml-stats**: enhance existing ML implementations ([237e02e](https://github.com/urmzd/supersource/commit/237e02e79b5f88e27ba3e4d6bd27c0f75a5a5a87))
- **ml-stats**: update assignment references and framework code ([ab9cc18](https://github.com/urmzd/supersource/commit/ab9cc1835377586ae034f4dfc582bf111218de0c))
- **interview**: categorize each section ([121fca5](https://github.com/urmzd/supersource/commit/121fca57734b6438b20b8e189c1c28ef55c79589))
- rename example-8.py to example-8.c ([2164e3d](https://github.com/urmzd/supersource/commit/2164e3df56e00ff9256ec06447f6133b7b2359b2))

### Miscellaneous

- auto-format Python files with ruff ([05e7c9a](https://github.com/urmzd/supersource/commit/05e7c9a9a3be0c866f79c4aff68fa5bf34085723))
- standardize CI/CD — add sr.yaml, ci.yml, release workflow ([d8b2505](https://github.com/urmzd/supersource/commit/d8b25050910588b3354028294a3645b8cc35d863))
- **practice**: remove incomplete practice implementations ([4a354da](https://github.com/urmzd/supersource/commit/4a354daffe5b0130d21b9e97e1ccc7598dbd6f04))
- add sensitive paths to .gitignore ([73358a5](https://github.com/urmzd/supersource/commit/73358a58683a0b352a061d879bb33b9251942986))
- license under Apache 2.0 ([b4168d6](https://github.com/urmzd/supersource/commit/b4168d64b3b13113b74bc4435239c4243d9b18fc))
- add python version specification and project configuration ([510f303](https://github.com/urmzd/supersource/commit/510f303f6fec29441cb665484f70860dd817d406))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.1.0...v0.2.0)

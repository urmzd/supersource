# Changelog

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

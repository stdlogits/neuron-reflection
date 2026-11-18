#!/bin/bash
# Rebuild the records in data/summary/ from scratch, in dependency order: the greedy orders before the runs that
# edit their first N units, the runs before the judge and the analyses that read them, then the figures. Each command
# overwrites its record. On one 24 GB GPU this takes a day or so; run any line on its own to rebuild its record, once
# the orders it reads exist. Not rebuilt: judge-validation.json (replies labelled blind by Claude, embedded in it) and
# exploratory/ (findings §2-§5's outputs from scripts that preceded the package).
# Generated from the records themselves (contrast, suite, conditions), so each command asks what its record holds;
# repro-mother-father's and sparse-fresh's also add the conditions of findings §2-§5, which earlier scripts measured.
# §1-§24's records come from an Apple-silicon Mac with locally fitted lenses; rerun on CUDA with the released lenses
# they come out close but not identical (findings.md, top).
set -e
cd "$(dirname "$0")/.."

# weights only (CPU): findings §1
python scripts/pole_reads.py

# greedy orders by effect: neurons (§17), refusal openers (§20), heads (§29), mixed (§30), fusion (§31), splits (§33)
python scripts/sparsity.py --contrasts mother-father ai-human dog-python
python scripts/order.py --contrast ai-cat --edit 'word[all]' --objective ai-cat-fuse --mix nqkvo --upto 200 --name fuse-mix-ai-cat
python scripts/order.py --contrast ai-cat --edit 'word[all]' --objective ai-cat-fuse --mix nqkvo --k 1 --upto 200 --name fuse-mixp-ai-cat
python scripts/order.py --contrast ai-cat --edit 'word[all]' --objective ai-cat-swap-lookup --heads k --upto 64 --name heads-ai-cat-k
python scripts/order.py --contrast ai-cat --edit 'word[all]' --objective ai-cat-swap-role --heads o --upto 64 --name heads-ai-cat-o
python scripts/order.py --contrast ai-cat --edit 'word[all]' --objective ai-cat-swap-lookup --heads q --upto 64 --name heads-ai-cat-q
python scripts/order.py --contrast ai-cat --edit 'word[all]' --objective ai-cat-swap-role --heads v --upto 64 --name heads-ai-cat-v
python scripts/order.py --contrast mother-father --edit 'word[all]' --objective mother-father-swap-role --heads o --upto 64 --name heads-mother-father-o
python scripts/order.py --contrast mother-father --edit 'word[all]' --objective mother-father-swap-lookup --heads q --upto 64 --name heads-mother-father-q
python scripts/order.py --contrast ai-cat --edit 'word[all]' --objective ai-cat-exchange --mix nqkvo --upto 200 --name mix-ai-cat
python scripts/order.py --contrast ai-cat --edit 'word[all]' --objective ai-cat-exchange --mix qkvo --upto 200 --name mixh-ai-cat
python scripts/order.py --contrast ai-cat --edit 'word[all]' --objective ai-cat-exchange --mix n --upto 200 --name mixn-ai-cat
python scripts/order.py --contrast harm-benign --edit 'template[en]' --objective refuse-frames --upto 300 --name refuse-frames
python scripts/order.py --contrast black-orange-cat --edit 'word[all]' --objective cat-split --mix nqkvo --k -2 --upto 200 --name split-amp-cat
python scripts/order.py --contrast black-orange-cat --edit 'word[all]' --objective cat-split --mix nqkvo --upto 200 --name split-ref-cat

# runs
python scripts/run.py --contrast cat-home-fox-after --suite cat-homes-garfield --variants 'linpairs[en]/reads' 'linpairs[en]/reads & linpairs[en]#qkv' --name after-fox
python scripts/run.py --contrast cat-home-garfield-after --suite cat-homes-garfield --variants 'linpairs[en]/reads' 'linpairs[en]/reads & linpairs[en]#qkv' --name after-garfield
python scripts/run.py --contrast ai-cat --suite persona-cat --variants 'word[all]+lens[all]' 'word[all]+lens[all]@persona' 'template[all]' 'template[all]@persona' 'word[all]+lens[all] & template[all]' 'word[all]+lens[all] & template[all]@top+persona' --name ai-cat-template
python scripts/run.py --contrast ai-cat --suite persona-cat --variants 'lens[all]' 'word[all]' 'word[all]+lens[all]' --name ai-cat
python scripts/run.py --contrast ai-human --suite persona-human --variants 'lens[all]' 'word[all]' 'word[all]+lens[all]' --name ai-human
python scripts/run.py --contrast animals-languages --suite animals-languages --variants 'category[all]@top:0.003' 'category[all]@top:0.03' 'category[all]@top:0.1' 'pairs[all]+lenspairs[all]@top:0.003' 'pairs[all]+lenspairs[all]@top:0.03' 'pairs[all]+lenspairs[all]@top:0.1' --name animals-languages-sweep --no-baseline
python scripts/run.py --contrast animals-languages --suite animals-languages --variants 'category[all]' 'category[all]@top:1' 'pairs[all]' 'pairs[all]@top:1' 'pairs[all]+lenspairs[all]' 'pairs[all]+lenspairs[all]@top:1' --name animals-languages
python scripts/run.py --contrast black-orange-cat --suite cat-split --variants 'word[all]@top:1^-0.5 & word[all]#qkvo^-0.5' 'word[all]@top:1^-1 & word[all]#qkvo^-1' 'word[all]@top:1^-2 & word[all]#qkvo^-2' 'word[all]^-2' --name cat-split
python scripts/run.py --contrast colors-genders --suite colors-genders --variants 'category[all]' 'category[all]@top:0.03' 'category[all]+lenscategory[all]' 'category[all]+lenscategory[all]@top:0.03' --name colors-genders
python scripts/run.py --contrast blue-split --suite colors-split --variants --name colors-split
python scripts/run.py --contrast ai-cat --suite ai-cat-dial --variants 'word[all]^0.5' 'word[all]^1' 'word[all]^1.5' 'word[all]^2' 'word[all]^2.5' 'word[all]^3' 'word[all]^4' --name dial-ai-cat
python scripts/run.py --contrast mother-father --suite mother-father-dial --variants 'lens[en,zh]^0.5' 'lens[en,zh]^1' 'lens[en,zh]^1.5' 'lens[en,zh]^2' 'lens[en,zh]^2.5' 'lens[en,zh]^3' 'lens[en,zh]^4' --name dial-mother-father
python scripts/run.py --contrast dog-python --suite dog-python --variants 'lens[all]' 'word[all]' 'word[all]+lens[all]' --name dog-python
python scripts/run.py --contrast ai-cat --suite persona-cat --variants 'word[all]+lens[all]/steer=0.1' 'word[all]+lens[all]/steer=0.25' 'word[all]+lens[all]/steer=0.5' 'word[all]+lens[all]@top:0.001/steer' 'word[all]+lens[all]@top:0.001/writes' --name dose-ai-cat --no-baseline
python scripts/run.py --contrast mother-father --suite parents --variants 'word[all]+lens[all]/steer=0.1' 'word[all]+lens[all]/steer=0.25' 'word[all]+lens[all]@top:0.001/steer' --name dose-mother-father --no-baseline
python scripts/run.py --contrast ai-human --suite eval-ai-human --variants 'word[all]+lens[all]' 'lens[all]' 'random[1]' 'random[2]' 'word[all]+lens[all]@random1' 'word[all]+lens[all]@random2' --name eval-ai-human
python scripts/run.py --contrast dog-python --suite eval-dog-python --variants 'word[all]+lens[all]' 'word[all]' 'random[1]' 'random[2]' 'word[all]+lens[all]@random1' 'word[all]@random1' --name eval-dog-python
python scripts/run.py --contrast mother-father --suite eval-mother-father --variants 'lens[en,zh]' 'word[all]+lens[all]' 'random[1]' 'random[2]' 'lens[en,zh]@random1' 'lens[en,zh]@random2' 'word[all]+lens[all]@random1' --name eval-mother-father
python scripts/run.py --contrast cat-home-fox-forms --suite cat-homes-forms --variants 'linpairs[en]/reads' 'linpairs[en]/reads & linpairs[en]#qkv' --name forms-fox-forms
python scripts/run.py --contrast cat-home-fox --suite cat-homes-forms --variants 'linpairs[en]/reads' 'linpairs[en]/reads & linpairs[en]#qkv' --name forms-fox
python scripts/run.py --contrast framed-plain --suite frames --variants 'template[grandma,fiction,nowarn]' 'template[grandma,fiction,nowarn]/reads' 'template[grandma,fiction,nowarn]/writes' 'template[evil,dan,devmode]' 'template[all]' 'template[grandma,fiction,nowarn]@random' --name frames-fusion
python scripts/run.py --contrast framed-plain --suite frames-heldout --variants 'template[grandma,fiction,nowarn]^3/reads' 'template[grandma,fiction,nowarn]^-1/writes' 'template[grandma,fiction,nowarn]^2/reads & template[grandma,fiction,nowarn]^-1/writes' 'template[grandma,fiction,nowarn]^3/reads & template[grandma,fiction,nowarn]^-1/writes' 'template[grandma,fiction,nowarn]@random^3/reads & template[grandma,fiction,nowarn]@random^-1/writes' --name frames-heldout
python scripts/run.py --contrast framed-plain --suite frames --variants 'template[grandma,fiction,nowarn]^1/reads' 'template[grandma,fiction,nowarn]^3/reads' 'template[grandma,fiction,nowarn]@random/reads' 'template[grandma,fiction,nowarn]^-1/writes' 'template[grandma,fiction,nowarn]^1/writes' 'template[grandma,fiction,nowarn]^-1' 'template[grandma,fiction,nowarn]^1' 'template[grandma,fiction,nowarn]@random^-1' --name frames-k
python scripts/run.py --contrast ai-cat --suite ai-cat-heads --variants 'word[all]@top:1^1' 'word[all]#qkvo^1' 'word[all]@top:1^1 & word[all]#qkvo^1' 'word[all]@top:1^0.5 & word[all]#qkvo^0.5' 'word[all]@top:1^1.5 & word[all]#qkvo^1.5' 'word[all]@top:1 & word[all]#qkvo' 'word[all]@top:1^1/reads & word[all]#qkv^1' 'word[all]@top:1^1/writes & word[all]#o^1' 'word[all]#qk^1' 'word[all]#ov^1' --name fuse-ai-cat
python scripts/run.py --contrast mother-father --suite mother-father-heads --variants 'word[all]@top:1^1' 'word[all]#qkvo^1' 'word[all]@top:1^1 & word[all]#qkvo^1' 'word[all]@top:1^0.5 & word[all]#qkvo^0.5' 'word[all]@top:1^1.5 & word[all]#qkvo^1.5' 'word[all]@top:1 & word[all]#qkvo' 'word[all]@top:1^1/reads & word[all]#qkv^1' 'word[all]@top:1^1/writes & word[all]#o^1' 'word[all]#qk^1' 'word[all]#ov^1' --name fuse-mother-father
python scripts/run.py --contrast ai-cat --suite ai-cat-heads --variants 'word[all]' 'word[all]#ov' 'word[all]#qk' 'word[all]#q' 'word[all] & word[all]#ov' 'word[all] & word[all]#qk' 'word[all] & word[all]#q' 'word[all]@top:1 & word[all]#qkvo' --name heads-ai-cat
python scripts/run.py --contrast ai-cat --suite ai-cat-heads --variants 'word[all]^1' 'word[all]^3' --name heads-dial-ai-cat
python scripts/run.py --contrast mother-father --suite mother-father-heads --variants 'lens[en,zh]^1' 'lens[en,zh]^3' 'lens[en,zh]^4' --name heads-dial-mother-father
python scripts/run.py --contrast mother-father --suite mother-father-heads --variants 'lens[en,zh]' 'lens[en,zh]#ov' 'lens[en,zh]#qk' 'lens[en,zh]#q' 'lens[en,zh] & lens[en,zh]#ov' 'lens[en,zh] & lens[en,zh]#qk' 'lens[en,zh] & lens[en,zh]#q' 'lens[en,zh]#qk^1' 'lens[en,zh] & lens[en,zh]#qk^1' 'lens[en,zh]@top:1 & lens[en,zh]#qkvo' --name heads-mother-father
python scripts/run.py --contrast ai-cat --suite ai-cat-heads --variants 'word[all]#o' 'word[all]#v' 'word[all]#k' 'word[all] & word[all]#o' 'word[all] & word[all]#v' --name heads-sides-ai-cat
python scripts/run.py --contrast mother-father --suite mother-father-heads --variants 'lens[en,zh]#o' 'lens[en,zh]#v' 'lens[en,zh]#k' 'lens[en,zh] & lens[en,zh]#o' 'lens[en,zh] & lens[en,zh]#v' --name heads-sides-mother-father
python scripts/run.py --contrast ai-cat --suite ai-cat-heads --variants 'word[all]#qkvo' 'word[all]@top:1' 'word[all] & word[all]#qkvo' --name heads-whole-ai-cat
python scripts/run.py --contrast mother-father --suite mother-father-heads --variants 'word[all]#qkvo' 'word[all]@top:1' 'word[all] & word[all]#qkvo' --name heads-whole-mother-father
python scripts/run.py --contrast mother-father --suite mother-father-heads --variants 'word[all]' 'word[all]#q' 'word[all]#ov' 'word[all] & word[all]#ov' 'word[all]#o' 'word[all] & word[all]#o' 'word[all]@top:1 & word[all]#qkvo' --name heads-word-mother-father
python scripts/run.py --contrast cat-home-bengal --suite cat-homes-bengal --variants 'linpairs[en]/reads' 'linpairs[en]/reads & linpairs[en]#qkv' --name home-bengal
python scripts/run.py --contrast cat-home-caracal --suite cat-homes --variants 'linpairs[en]/reads' 'linpairs[en]@top:0.1/reads' 'linpairs[en]/reads & linpairs[en]#qkv' 'linpairs[en]@top:0.1/reads & linpairs[en]#qkv' --name home-caracal
python scripts/run.py --contrast cat-home-fox --suite cat-homes --variants 'linpairs[en]/reads' 'linpairs[en]@top:0.1/reads' 'linpairs[en]/reads & linpairs[en]#qkv' 'linpairs[en]@top:0.1/reads & linpairs[en]#qkv' --name home-fox
python scripts/run.py --contrast cat-home-garfield --suite cat-homes-garfield --variants 'linpairs[en]/reads' 'linpairs[en]/reads & linpairs[en]#qkv' --name home-garfield
python scripts/run.py --contrast cat-home-vorrel --suite cat-homes --variants 'linpairs[en]/reads' 'linpairs[en]@top:0.1/reads' 'linpairs[en]/reads & linpairs[en]#qkv' 'linpairs[en]@top:0.1/reads & linpairs[en]#qkv' --name home-vorrel
python scripts/run.py --contrast ai-cat --suite ai-cat-inside --variants 'lens[all]' 'word[all]' 'word[all]+lens[all]' 'word[all]+lens[all] & template[all]' --name inside-ai-cat
python scripts/run.py --contrast animals-languages --suite animals-languages-inside --variants 'category[all]' 'category[all]@top:0.03' 'pairs[all]+lenspairs[all]@top:0.03' --name inside-animals-languages
python scripts/run.py --contrast colors-genders --suite colors-genders-inside --variants 'category[all]@top:0.03' 'category[all]+lenscategory[all]@top:0.03' 'hardpairs[all]+lenshardpairs[all]' --name inside-colors-genders
python scripts/run.py --contrast colors-genders-nb --suite colors-genders-nb --variants 'softpairs[all]' 'softpairs[all]@top:0.03' 'hardpairs[all]' 'category[all]' --name model-pairs-colors-genders-nb --no-baseline
python scripts/run.py --contrast colors-genders --suite colors-genders-nb --variants 'softpairs[all]' 'softpairs[all]+lenssoftpairs[all]' 'softpairs[all]+lenssoftpairs[all]@top:0.03' 'hardpairs[all]+lenshardpairs[all]' --name model-pairs-colors-genders
python scripts/run.py --contrast mother-volcano --suite orthogonal --variants 'lens[all]' 'word[all]' 'word[all]+lens[all]' --name mother-volcano
python scripts/run.py --contrast harm-benign --suite judgment --variants 'lens[harm]' 'lens[harm]/reads' 'lens[harm]/writes' 'template[en]/writes' 'template[en]/reads' 'lens[refuse]' --name refusal-judgment
python scripts/run.py --contrast harm-benign --suite refusal --variants 'template[en]' 'template[en]/reads' 'template[en]/writes' 'lens[refuse]' 'template[en]+lens[refuse]' 'lens[harm]' 'template[en]@random' 'random[1]' --name refusal-sides
python scripts/run.py --contrast mother-father --suite parents --variants 'lens[en,zh]' 'word[all]+lens[all]' 'lens[en,zh]@top:1' 'lens[en,zh]^1' 'lens[en,zh]@top:1^1' 'lens[all]' 'word[en,zh,de,fr,es]' 'word[en,zh,de,fr,es]+lens[en,zh]' 'word[all]' --name repro-mother-father
python scripts/run.py --contrast ai-cat --suite ai-cat-robot --variants 'word[all]' 'lens[all]' 'word[all]+lens[all]' --name robot-ai-cat
python scripts/run.py --contrast ai-cat --suite persona-cat --variants 'word[all]+lens[all]' 'word[all]' 'template[all]' --name shares-ai-cat --no-baseline --edits-only
python scripts/run.py --contrast dog-python --suite dog-python --variants 'lens[all]' 'word[all]' 'word[all]+lens[all]' --name shares-dog-python --no-baseline --edits-only
python scripts/run.py --contrast mother-father --suite parents --variants 'lens[en,zh]' 'word[all]+lens[all]' 'word[all]' --name shares-mother-father --no-baseline --edits-only
python scripts/run.py --contrast mother-volcano --suite orthogonal --variants 'word[all]' 'word[all]+lens[all]' --name shares-mother-volcano --no-baseline --edits-only
python scripts/run.py --contrast ai-cat --suite persona-cat --variants 'word[all]+lens[all]' 'word[all]+lens[all]/reads' 'word[all]+lens[all]/writes' --name sides-ai-cat
python scripts/run.py --contrast dog-python --suite dog-python --variants 'word[all]' 'word[all]/reads' 'word[all]/writes' 'word[all]+lens[all]' 'word[all]+lens[all]/reads' 'word[all]+lens[all]/writes' --name sides-dog-python
python scripts/run.py --contrast mother-father --suite parents --variants 'lens[en,zh]' 'lens[en,zh]/reads' 'lens[en,zh]/writes' 'word[all]+lens[all]' 'word[all]+lens[all]/reads' 'word[all]+lens[all]/writes' --name sides-mother-father
python scripts/run.py --contrast ai-cat --suite persona-cat --variants 'word[all]+lens[all]/steer' 'word[all]/steer' --name steer-ai-cat --no-baseline
python scripts/run.py --contrast dog-python --suite dog-python --variants 'word[all]+lens[all]/steer' 'word[all]/steer' --name steer-dog-python --no-baseline
python scripts/run.py --contrast mother-father --suite parents --variants 'lens[en,zh]/steer' 'word[all]+lens[all]/steer' 'word[all]/steer' --name steer-mother-father --no-baseline

# runs that edit the first units of an order
python scripts/run.py --contrast black-orange-cat --suite cat-split --variants 'word[all]@mix:16^-2' 'word[all]@mix:32^-2' 'word[all]@mix:64^-2' 'word[all]@mix:128^-2' 'word[all]@mix:200^-2' 'word[all]@rmix:200^-2' --name cat-split-amp --no-baseline --order split-amp-cat
python scripts/run.py --contrast black-orange-cat --suite cat-split --variants 'word[all]@mix:16' 'word[all]@mix:32' 'word[all]@mix:64' 'word[all]@mix:128' 'word[all]@mix:200' 'word[all]@rmix:200' --name cat-split-ref --no-baseline --order split-ref-cat
python scripts/run.py --contrast ai-cat --suite ai-cat-exchange --variants 'word[all]@mix:8' 'word[all]@mix:16' 'word[all]@mix:32' 'word[all]@mix:64' 'word[all]@mix:128' 'word[all]@mix:200' 'word[all]@rmix:64' 'word[all]@rmix:200' --name exchange-mix-ai-cat --order mix-ai-cat
python scripts/run.py --contrast ai-cat --suite ai-cat-exchange --variants 'word[all]@mix:16' 'word[all]@mix:64' 'word[all]@mix:200' --name exchange-mixh-ai-cat --order mixh-ai-cat
python scripts/run.py --contrast ai-cat --suite ai-cat-exchange --variants 'word[all]@mix:16' 'word[all]@mix:64' 'word[all]@mix:200' --name exchange-mixn-ai-cat --order mixn-ai-cat
python scripts/run.py --contrast ai-cat --suite ai-cat-exchange --variants 'word[all]@mix:128^1' 'word[all]@mix:200^1' 'word[all]@mix:128^1.5' --name fuse-exchange-ai-cat --order mix-ai-cat
python scripts/run.py --contrast ai-cat --suite ai-cat-exchange --variants 'word[all]@mix:16' 'word[all]@mix:32' 'word[all]@mix:64' 'word[all]@mix:128' 'word[all]@mix:200' 'word[all]@rmix:200' --name fuse-sparse-ai-cat --order fuse-mix-ai-cat
python scripts/run.py --contrast ai-cat --suite ai-cat-exchange --variants 'word[all]@mix:16^1' 'word[all]@mix:32^1' 'word[all]@mix:64^1' 'word[all]@mix:128^1' 'word[all]@mix:200^1' 'word[all]@rmix:200^1' --name fuse-sparsep-ai-cat --order fuse-mixp-ai-cat
python scripts/run.py --contrast mother-father --suite mother-father-inside --variants 'lens[en,zh]' 'lens[en,zh]@greedy:30' 'word[all]' 'word[all]+lens[all]' --name inside-mother-father
python scripts/run.py --contrast harm-benign --suite frames --variants 'template[en]@greedy:10' 'template[en]@greedy:30' 'template[en]@greedy:100' 'template[en]@rgreedy:30' 'template[en]@rgreedy:100' --name patch-refuse-frames --order refuse-frames
python scripts/run.py --contrast ai-human --suite refusal --variants 'word[all]+lens[all]' 'word[all]+lens[all]@greedy:100' 'lens[all]' --name refusal-ai-human
python scripts/run.py --contrast dog-python --suite dog-python --variants 'word[all]+lens[all]@greedy:30' 'word[all]+lens[all]@greedy:100' 'word[all]+lens[all]@greedy:300' 'word[all]+lens[all]' --name sparse-dog-python
python scripts/run.py --contrast mother-father --suite fresh --variants 'lens[en,zh]@greedy:100' 'lens[en,zh]@greedy:300' 'lens[en,zh]' 'word[all]' 'word[all]+lens[all]' --name sparse-fresh
python scripts/run.py --contrast ai-cat --suite ai-cat-sparse --variants 'word[all]@greedy:1#k' 'word[all]@greedy:2#k' 'word[all]@greedy:4#k' 'word[all]@greedy:8#k' 'word[all]@greedy:16#k' 'word[all]@greedy:32#k' 'word[all]@greedy:64#k' 'word[all]@rgreedy:16#k' 'word[all]@loading:16#k' 'word[all]@rgreedy:64#k' 'word[all]@loading:64#k' --name sparse-heads-ai-cat-k --order heads-ai-cat-k
python scripts/run.py --contrast ai-cat --suite ai-cat-sparse --variants 'word[all]@greedy:1#o' 'word[all]@greedy:2#o' 'word[all]@greedy:4#o' 'word[all]@greedy:8#o' 'word[all]@greedy:16#o' 'word[all]@greedy:32#o' 'word[all]@greedy:64#o' 'word[all]@rgreedy:16#o' 'word[all]@loading:16#o' 'word[all]@rgreedy:64#o' 'word[all]@loading:64#o' --name sparse-heads-ai-cat-o --order heads-ai-cat-o
python scripts/run.py --contrast ai-cat --suite ai-cat-sparse --variants 'word[all]@greedy:1#q' 'word[all]@greedy:2#q' 'word[all]@greedy:4#q' 'word[all]@greedy:8#q' 'word[all]@greedy:16#q' 'word[all]@greedy:32#q' 'word[all]@greedy:64#q' 'word[all]@rgreedy:16#q' 'word[all]@loading:16#q' 'word[all]@rgreedy:64#q' 'word[all]@loading:64#q' --name sparse-heads-ai-cat-q --order heads-ai-cat-q
python scripts/run.py --contrast ai-cat --suite ai-cat-sparse --variants 'word[all]@greedy:1#v' 'word[all]@greedy:2#v' 'word[all]@greedy:4#v' 'word[all]@greedy:8#v' 'word[all]@greedy:16#v' 'word[all]@greedy:32#v' 'word[all]@greedy:64#v' 'word[all]@rgreedy:16#v' 'word[all]@loading:16#v' 'word[all]@rgreedy:64#v' 'word[all]@loading:64#v' --name sparse-heads-ai-cat-v --order heads-ai-cat-v
python scripts/run.py --contrast mother-father --suite mother-father-sparse --variants 'word[all]@greedy:1#o' 'word[all]@greedy:2#o' 'word[all]@greedy:4#o' 'word[all]@greedy:8#o' 'word[all]@greedy:16#o' 'word[all]@greedy:32#o' 'word[all]@greedy:64#o' 'word[all]@rgreedy:16#o' 'word[all]@loading:16#o' 'word[all]@rgreedy:64#o' 'word[all]@loading:64#o' --name sparse-heads-mother-father-o --order heads-mother-father-o
python scripts/run.py --contrast mother-father --suite mother-father-sparse --variants 'word[all]@greedy:1#q' 'word[all]@greedy:2#q' 'word[all]@greedy:4#q' 'word[all]@greedy:8#q' 'word[all]@greedy:16#q' 'word[all]@greedy:32#q' 'word[all]@greedy:64#q' 'word[all]@rgreedy:16#q' 'word[all]@loading:16#q' 'word[all]@rgreedy:64#q' 'word[all]@loading:64#q' --name sparse-heads-mother-father-q --order heads-mother-father-q
python scripts/run.py --contrast mother-father --suite parents --variants 'lens[en,zh]@greedy:30' 'lens[en,zh]@greedy:100' 'lens[en,zh]@greedy:300' 'lens[en,zh]' --name sparse-parents
python scripts/run.py --contrast ai-human --suite persona-human --variants 'word[all]+lens[all]@greedy:30' 'word[all]+lens[all]@greedy:100' 'word[all]+lens[all]@greedy:300' 'word[all]+lens[all]' --name sparse-persona-human

# refusal labels from Gemma-4-12B-it (24 GB in bf16, run alone), written into the six records in place
python scripts/judge.py refusal-sides refusal-ai-human frames-fusion frames-k frames-heldout patch-refuse-frames

# analyses that write records
python scripts/correspondence.py --contrast animals-languages
python scripts/correspondence.py --contrast colors-genders-nb
python scripts/correspondence.py --contrast colors-genders
python scripts/drift.py --contrast ai-cat --axes 'word[all]'
python scripts/drift.py --contrast mother-father --axes 'lens[en,zh]'
python scripts/drift.py --contrast mother-father --axes 'word[all]'
python scripts/off_axis.py --contrast ai-cat
python scripts/off_axis.py --contrast mother-father
python scripts/offset.py --contrast ai-cat --axes 'word[all]' --suite ai-cat-heads --heads
python scripts/offset.py --contrast ai-cat --axes 'word[all]' --suite ai-cat-heads
python scripts/offset.py --contrast mother-father --axes 'lens[en,zh]' --suite mother-father-heads --heads
python scripts/offset.py --contrast mother-father --axes 'word[all]' --suite mother-father-heads --heads
python scripts/offset.py --contrast mother-father --axes 'lens[en,zh]' --suite mother-father-heads
python scripts/offset.py --contrast mother-father --axes 'word[all]' --suite mother-father-heads
python scripts/ripple.py

# figures from the records
python scripts/figures.py
python scripts/charts.py
if [ -d node_modules/pele ]; then node docs/figures/render.mjs; else echo "charts not redrawn: npm install pele"; fi
python scripts/graphics.py

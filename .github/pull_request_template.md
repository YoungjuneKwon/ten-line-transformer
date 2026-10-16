<!--
This file is 123 lines and exists to be read, not to grow. So the three
questions below are the whole review.

Not looking for: biases, dropout, config classes, flash attention, CLI
frameworks, type annotations, a test directory, or anything that splits the
file into modules. Those are all improvements to a different repository.
-->

**What is wrong today**

<!-- A bug, a line that does the same thing less clearly, a claim that does not
     match the code. If this is a correction to NOTICE.md or line-mapping.md,
     link the line that contradicts it. -->

**What you ran, and what it printed**

```
$ OMP_NUM_THREADS=1 python scripts/check_reproduction.py

```

<!-- If the diff touches transformer.py, the loss values must still match. If
     they are supposed to change, say so here and say why. -->

**If this adds a line, what does the line buy**

<!-- Leave blank if the diff removes lines or leaves the count alone. -->

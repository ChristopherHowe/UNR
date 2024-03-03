| File                       | Operation | Type & Rounds | Time (no locking) | Time (with locking) |
|----------------------------|-----------|---------------|-------------------|---------------------|
| ten.txt                    | threaded  | 10000         | 0.257 ms          | 0.256 ms            |
| ten.txt                    | looped    | 10000         | 0.000 ms          |                     |
| oneThousand.txt            | threaded  | 10000         | 0.479 ms          | 0.464 ms            |
| oneThousand.txt            | looped    | 10000         | 0.002 ms          |                     |
| 100000-random-numbers.txt   | threaded  | 1000          | 0.553 ms          | 0.538 ms            |
| 100000-random-numbers.txt   | looped    | 1000          | 0.215 ms          |                     |
| 10000000-random-numbers.txt | threaded  | 100           | 5.447 ms          | 5.739 ms            |
| 10000000-random-numbers.txt | looped    | 50            | 20.426 ms         |                     |

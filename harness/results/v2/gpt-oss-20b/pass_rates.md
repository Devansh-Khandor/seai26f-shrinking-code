# Pass rates — model `gpt-oss-20b`

Per skill attempt (both requests must pass for semantic). Every attempt stays in the denominator; `semantic (verified)` divides by attempts the checker could judge (pass + fail) instead. `contract only` = passed the original format contract but not the semantic check.

| domain | treatment | steps | runs | contract | semantic | semantic (verified) | contract only | unverified | checker errors | crashed | parse_failed | overflow | truncated | timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apache | T1 | 30 | 90% | 67% | 33% | 37% of 27 | 11 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| apache | T2 | 30 | 93% | 87% | 43% | 46% of 28 | 13 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| apache | T3 | 30 | 87% | 80% | 63% | 73% of 26 | 7 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| apache | T4 | 30 | 90% | 70% | 53% | 59% of 27 | 9 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| apache | T5 | 30 | 87% | 77% | 73% | 85% of 26 | 3 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| apache | T6 | 30 | 43% | 30% | 37% | 85% of 13 | 1 | 0 | 0 | 15 | 2 | 0 | 0 | 0 |
| cars | T1 | 30 | 90% | 77% | 47% | 52% of 27 | 11 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| cars | T2 | 30 | 70% | 53% | 33% | 48% of 21 | 7 | 0 | 0 | 8 | 1 | 0 | 0 | 0 |
| cars | T3 | 30 | 80% | 70% | 50% | 62% of 24 | 8 | 0 | 0 | 5 | 1 | 0 | 0 | 0 |
| cars | T4 | 30 | 73% | 67% | 33% | 45% of 22 | 10 | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| cars | T5 | 30 | 77% | 67% | 37% | 48% of 23 | 12 | 0 | 0 | 7 | 0 | 0 | 0 | 0 |
| cars | T6 | 30 | 23% | 17% | 17% | 71% of 7 | 1 | 0 | 0 | 21 | 2 | 0 | 0 | 0 |
| nasa | T1 | 30 | 77% | 73% | 30% | 39% of 23 | 14 | 0 | 0 | 7 | 0 | 0 | 0 | 0 |
| nasa | T2 | 30 | 87% | 80% | 17% | 19% of 26 | 19 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| nasa | T3 | 30 | 63% | 47% | 17% | 26% of 19 | 10 | 0 | 0 | 11 | 0 | 0 | 0 | 0 |
| nasa | T4 | 30 | 67% | 57% | 13% | 20% of 20 | 14 | 0 | 0 | 7 | 3 | 0 | 0 | 0 |
| nasa | T5 | 30 | 77% | 60% | 27% | 35% of 23 | 10 | 0 | 0 | 7 | 0 | 0 | 0 | 0 |
| nasa | T6 | 30 | 50% | 37% | 10% | 20% of 15 | 9 | 0 | 0 | 13 | 2 | 0 | 0 | 0 |

## By foundation (all domains)

| domain | treatment | steps | runs | contract | semantic | semantic (verified) | contract only | unverified | checker errors | crashed | parse_failed | overflow | truncated | timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | plain Python (T1+T2) | 180 | 84% | 73% | 34% | 40% of 152 | 75 | 0 | 0 | 26 | 2 | 0 | 0 | 0 |
| all | scikit-learn (T3+T4) | 180 | 77% | 65% | 38% | 50% of 138 | 58 | 0 | 0 | 38 | 4 | 0 | 0 | 0 |
| all | ezr (T5+T6) | 180 | 59% | 48% | 33% | 56% of 107 | 36 | 0 | 0 | 67 | 6 | 0 | 0 | 0 |

## Failure categories

`system:*` and `evaluator:*` are the machine's or the checker's problem, not the model's.

| domain | treatment | categories |
|---|---|---|
| apache | T1 | code:format 1, code:runtime 3, code:semantic 17, none (passed) 9 |
| apache | T2 | code:no_answer 1, code:runtime 2, code:semantic 14, none (passed) 13 |
| apache | T3 | code:format 2, code:runtime 3, code:semantic 7, code:syntax 1, none (passed) 17 |
| apache | T4 | code:format 4, code:runtime 3, code:semantic 11, none (passed) 12 |
| apache | T5 | code:format 2, code:runtime 3, code:semantic 4, code:syntax 1, none (passed) 20 |
| apache | T6 | code:api_misuse 1, code:format 3, code:runtime 14, code:semantic 2, generation:extraction 2, none (passed) 8 |
| cars | T1 | code:format 2, code:no_answer 1, code:runtime 2, code:semantic 12, code:syntax 1, none (passed) 12 |
| cars | T2 | code:format 1, code:runtime 7, code:semantic 11, code:syntax 1, generation:extraction 1, none (passed) 9 |
| cars | T3 | code:api_misuse 1, code:format 2, code:runtime 4, code:semantic 9, generation:extraction 1, none (passed) 13 |
| cars | T4 | code:runtime 7, code:semantic 12, code:syntax 1, none (passed) 10 |
| cars | T5 | code:format 3, code:runtime 5, code:semantic 12, code:syntax 2, none (passed) 8 |
| cars | T6 | code:api_misuse 1, code:format 1, code:runtime 18, code:semantic 2, code:syntax 2, generation:extraction 2, none (passed) 4 |
| nasa | T1 | code:format 1, code:runtime 7, code:semantic 14, none (passed) 8 |
| nasa | T2 | code:runtime 3, code:semantic 21, generation:extraction 1, none (passed) 5 |
| nasa | T3 | code:api_misuse 6, code:format 1, code:runtime 5, code:semantic 14, none (passed) 4 |
| nasa | T4 | code:format 1, code:fragile 1, code:runtime 6, code:semantic 15, code:syntax 1, generation:extraction 3, none (passed) 3 |
| nasa | T5 | code:runtime 5, code:semantic 15, code:syntax 2, none (passed) 8 |
| nasa | T6 | code:format 1, code:runtime 11, code:semantic 12, code:syntax 2, generation:extraction 2, none (passed) 2 |

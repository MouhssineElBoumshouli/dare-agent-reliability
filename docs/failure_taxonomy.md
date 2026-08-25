# Failure Taxonomy

The original DARE-Bench paper discusses instruction-adherence errors, code errors, execution-limit failures, and maximum-token-limit failures. This study keeps those categories and adds an output-format category because reliability analysis needs to distinguish execution from evaluation failures.

Use the first mechanically supported category that explains the failed run.

1. `tool_call_error`
   - malformed tool arguments
   - missing required files/tool parameters
   - provider/tool protocol failure attributable to the generated call

2. `code_error`
   - Python exception
   - invalid imports
   - KeyError/shape/type errors
   - generated program terminates unsuccessfully

3. `execution_timeout`
   - sandbox execution exceeds the configured limit

4. `max_turn_or_token_limit`
   - run ends because the configured agent-turn or token budget is exhausted before a valid prediction is produced

5. `malformed_prediction`
   - `prediction.csv` missing required identifier/target columns
   - row alignment/count mismatch
   - unreadable CSV

6. `instruction_deviation`
   - code executes and output is structurally valid, but inspection shows a required v1 workflow step/parameter was omitted or changed

7. `wrong_prediction_unclassified`
   - structurally valid prediction fails exact match and no stronger category above is demonstrated

8. `infrastructure_error`
   - API outage, local Docker crash, disk failure, or other failure not attributable to the agent

Rules:

- Never relabel a model failure as infrastructure merely because rerunning succeeds.
- Preserve the run ID and evidence used for every manual classification.
- If unsure, use `wrong_prediction_unclassified` rather than inventing a cause.

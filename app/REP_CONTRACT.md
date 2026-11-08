# Rep Semantics and Selection Contract

RepChain and the legacy segmenter remain independent algorithms. Their final
outputs are joined through the `vls-rep-selection-v1` contract.

## Counts

| Field | Meaning |
| --- | --- |
| `accepted_cycle` | Reportable cycles; equals `rep_count` |
| `complete_cycle` | Cycle with both top boundaries observed |
| `endpoint_partial_cycle` | Interior bottom with one observed endpoint top |
| `rejected_candidate` | Candidate not promoted to a reportable cycle |

`accepted_cycle = complete_cycle + endpoint_partial_cycle`.

## Selection modes

| Mode | Final cycle source | Top boundary | Bottom boundary |
| --- | --- | --- | --- |
| `primary` | RepChain | RepChain phase signal | RepChain phase signal |
| `boundary_refinement` | RepChain | RepChain phase signal | Legacy candidate anchor |
| `fallback` | Legacy candidates | RepChain phase peak reconstruction | Legacy candidate anchor |
| `none` | No accepted cycle | None | None |

`selected_source` is `repchain` for `primary` and
`boundary_refinement`, `legacy` for `fallback`, and `null` for `none`.

## Compatibility

Top-level `rep_count` is retained as an alias for
`rep_selection.counts.accepted_cycle`. The former
`complete_rep_count`, `endpoint_partial_rep_count`,
`candidate_sources`, `recovery`, `recovery_used`, and
`boundary_refinement_used` fields remain available. New consumers should
read `rep_selection` and each Rep's `provenance` object.

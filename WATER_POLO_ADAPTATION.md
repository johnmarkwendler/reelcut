# Water Polo Adaptation Status

## Configured

- Water-polo Roboflow workspace and forked Workflow.
- Five exact detector classes: player, referee, ball, goalie, goal.
- User-supplied detector identifier.
- Original Reelcut digit model retained.
- Cap-region digit filtering and cap-color sampling.
- Water-polo CLI preset and UI language.

## First live milestone

Use a short representative clip to validate:

1. Roboflow accepts the supplied detector identifier.
2. Workflow outputs remain named `tracked_players`, `ball`, and
   `goal_detections`.
3. `goalie` is tracked with players and canonicalized to goalkeeper.
4. Cap crops contain enough pixels for the soccer jersey-digit model to
   transfer at all.
5. Goal and ball confidence thresholds provide useful recall.

## Calibration after the first debug video

- Tune cap crop bounds and digit confidence using exported failed crops.
- Measure ID switches with Workflow CMC off and on; keep off unless measured
  evidence favors it.
- Tune minimum player-box height for above-water boxes.
- Tune ball/goal thresholds and goal-box persistence for splashing/occlusion.
- Replace soccer-derived involvement thresholds with water-polo measurements.

The reused digit model is a deliberate experiment, not an assumption that
jersey digits and cap digits have identical visual behavior.

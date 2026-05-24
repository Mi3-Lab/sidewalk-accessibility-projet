# User Study App Evaluation Plan

## Purpose

This study validates the paper's weakest external-validity point: route quality is currently evaluated by model-predicted passability, not by new mobility-aid user judgments on the proposed routes.

The app should make participation fast enough that users do not abandon the study. The central product requirement is:

> One clear accessibility decision per screen, with large controls, optional reasons, autosave, and an easy stop point after a short session.

## Research Questions

1. Can mobility-aid users complete accessibility judgments with low friction in a web app?
2. Do new user judgments agree with the learned per-aid soft-label distributions?
3. Do users prefer our risk-aware route over standard or ORS-style alternatives?
4. Which failure modes explain disagreement between model-predicted passability and user preference?

## Participant Groups

Primary groups:

- Manual wheelchair users
- Power wheelchair users
- Mobility scooter users
- Walker users
- Walking cane users

Optional groups:

- White cane users
- Crutch users
- People who walk without an aid but have mobility limitations
- Caregivers who regularly plan accessible routes

The first MVP should support all groups but report the five paper groups separately.

## Study Phases

### Phase 0: Internal Smoke Test

Target: 3-5 internal testers.

Goal:

- Verify image loading, autosave, export, keyboard navigation, and mobile layout.
- Verify that every trial produces the expected record fields.

Pass criteria:

- 100% of trials are saved locally.
- Exported JSON and CSV are valid.
- No participant needs verbal explanation after the tutorial screen.

### Phase 1: Accessibility Pilot

Target: 8-12 mobility-aid users.

Goal:

- Measure burden before recruiting broadly.
- Identify confusing images, route views, or button labels.

Main metrics:

- Completion rate for a 20-image + 5-route session.
- Median response time per image.
- Skip rate.
- Drop-off screen.
- Ease score.
- Free-text complaints.

Pass criteria:

- At least 80% complete the base session.
- Median image response time below 20 seconds.
- Skip rate below 25%.
- Mean ease score at least 4 on a 1-5 scale.

### Phase 2: Main Evaluation

Target:

- Minimum: 100 participants total.
- Strong: 150-200 participants total.
- Ideal: at least 25 participants in each of the five main mobility-aid groups.

Per participant:

- 20 image trials.
- 5 route-comparison trials.
- 3 usability questions.
- Optional "do 10 more" extension.

Expected data:

- 2,000-6,000 new image judgments.
- 500-2,000 route preferences.
- Enough per-aid route preference data to report confidence intervals.

## App Flow

### 1. Consent

Keep this short. The user should understand:

- This is a research study.
- They can skip any question.
- They can stop any time.
- No personally identifying information is required in the MVP.

### 2. Mobility Profile

Required:

- Primary mobility aid.
- Familiarity with sidewalk accessibility issues.

Optional:

- Route-planning concerns: curb ramps, slope, surface cracks, width, crossings, obstructions.

### 3. Tutorial

Show two example tasks:

- One image judgment.
- One route comparison.

No scoring. The tutorial exists only to teach the interface.

### 4. Image Judgments

Prompt:

> Could you pass through this sidewalk segment using your mobility aid?

Responses:

- Yes
- Unsure
- No
- Skip

Optional reason chips:

- Curb or ramp issue
- Surface or cracks
- Too narrow
- Slope
- Obstruction
- Not enough information
- Other

### 5. Route Comparisons

Prompt:

> Which route would you choose for this trip?

Responses:

- Route A
- Route B
- No preference
- Neither route
- Skip

Optional reason chips:

- Shorter
- Safer surface
- Fewer crossings
- Avoids curb issue
- Avoids slope
- Looks more accessible
- Not enough information

Route labels should be randomized so participants do not know which route is ours.

### 6. Usability

Three 1-5 questions:

- The task was easy to understand.
- The images/routes gave enough information.
- I would be willing to use this kind of tool again.

Optional free text:

- What made any question hard to answer?

## Data Schema

### Participant Record

- `participant_id`
- `session_id`
- `created_at`
- `primary_aid`
- `experience_level`
- `concerns`
- `device_width`
- `user_agent`

### Image Trial Record

- `trial_id`
- `participant_id`
- `session_id`
- `trial_type = image`
- `image_id`
- `image_src`
- `mobility_aid`
- `response`
- `reason_chips`
- `response_time_ms`
- `zoom_used`
- `skipped`
- `created_at`

### Route Trial Record

- `trial_id`
- `participant_id`
- `session_id`
- `trial_type = route`
- `route_pair_id`
- `route_a_id`
- `route_b_id`
- `route_a_type`
- `route_b_type`
- `selected_route`
- `reason_chips`
- `response_time_ms`
- `skipped`
- `created_at`

### Usability Record

- `participant_id`
- `session_id`
- `ease_score`
- `information_score`
- `reuse_score`
- `free_text`
- `created_at`

## Analysis Plan

### App Feasibility

- Completion rate.
- Median time per image and route trial.
- Skip rate by trial type.
- Drop-off screen.
- Ease and information scores.

### New Human Labels

- Per-aid vote distributions.
- Vote entropy per image and per route pair.
- Agreement with original Project Sidewalk distributions when the same images are used.
- Consistency on repeated quality-control trials.

### Model Validation

- Soft Brier score of DINOv2-large Soft-KL against new labels.
- Soft Brier score of Hard-CE against new labels.
- Calibration curve for `p_yes`.
- Failure examples where model and users disagree.

### Routing Validation

Primary endpoint:

- Percentage of route comparisons where users prefer the risk-aware route.

Secondary endpoints:

- Preference rate by aid type.
- Preference rate when model-predicted passability gap is small vs large.
- Reasons selected for standard-route preference.
- "Neither route" rate.

Report Wilson 95% confidence intervals for preference proportions.

## MVP Implementation

The first app is intentionally static:

- `apps/accessibility-study/index.html`
- `apps/accessibility-study/styles.css`
- `apps/accessibility-study/app.js`
- `apps/accessibility-study/data/trials.json`

Capabilities:

- Works from a simple static server.
- Autosaves to `localStorage`.
- Supports resume.
- Supports JSON and CSV export.
- Supports keyboard navigation.
- Uses large mobile-first controls.

Backend can be added later after the pilot confirms the flow.

## Paper Integration

If the study succeeds, add a new section:

> Human Route-Preference Validation

Potential headline:

> In a low-burden web study with mobility-aid users, participants completed the base session in X minutes with Y% completion, and preferred the risk-aware route over the standard route in Z% of comparisons.

This directly addresses the current limitation that routing is evaluated by model-predicted passability rather than independent route-level user preference.

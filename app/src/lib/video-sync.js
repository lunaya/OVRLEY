import i18next from 'i18next'
import { getZonedDateTimeParts } from '@/lib/time-format'

const CORRELATION_COARSE_STEP_SECONDS = 1
const CORRELATION_FINE_STEP_SECONDS = 0.05
const CORRELATION_FINE_WINDOW_SECONDS = 2
const CORRELATION_MIN_OVERLAP_FLOOR_SECONDS = 10
const CORRELATION_MIN_OVERLAP_CAP_SECONDS = 60
const CORRELATION_MIN_SPEED_RANGE_MPS = 3
const CORRELATION_MIN_SPEED_STD_MPS = 0.5
const CORRELATION_MIN_SCORE = 0.85
const CORRELATION_MAX_MEAN_ABS_ERROR_MPS = 3.5

/**
 * Extracts finite (time, speed) pairs and validates that the series has enough
 * dynamic range to represent real movement rather than a stationary/cached fix.
 *
 * @param {object|null|undefined} activity Parsed activity object.
 * @returns {{times: number[], speeds: number[], duration: number}|null} Validated speed profile, or null.
 */
function extractSpeedProfile(activity) {
  const rawTimes = activity?.sample_elapsed_seconds
  const rawSpeeds = activity?.speed
  if (!Array.isArray(rawTimes) || !Array.isArray(rawSpeeds) || rawTimes.length < 2 || rawSpeeds.length !== rawTimes.length) {
    return null
  }

  const times = []
  const speeds = []
  let minSpeed = Number.POSITIVE_INFINITY
  let maxSpeed = Number.NEGATIVE_INFINITY
  let sum = 0
  let sumSq = 0

  for (let index = 0; index < rawTimes.length; index += 1) {
    const t = rawTimes[index]
    const s = rawSpeeds[index]
    if (typeof t !== 'number' || typeof s !== 'number' || !Number.isFinite(t) || !Number.isFinite(s)) continue
    times.push(t)
    speeds.push(s)
    if (s < minSpeed) minSpeed = s
    if (s > maxSpeed) maxSpeed = s
    sum += s
    sumSq += s * s
  }

  if (times.length < 2) return null
  const duration = times[times.length - 1] - times[0]
  if (duration < CORRELATION_MIN_OVERLAP_FLOOR_SECONDS) return null
  if (maxSpeed - minSpeed < CORRELATION_MIN_SPEED_RANGE_MPS) return null

  const mean = sum / speeds.length
  const variance = Math.max(0, sumSq / speeds.length - mean * mean)
  if (Math.sqrt(variance) < CORRELATION_MIN_SPEED_STD_MPS) return null

  return { times, speeds, duration }
}

/**
 * Resamples a irregularly or regularly spaced (times, speeds) profile onto a
 * uniform grid starting at t = 0 with spacing `stepSeconds`.
 *
 * @param {{times: number[], speeds: number[]}} profile Validated speed profile.
 * @param {number} stepSeconds Uniform sample interval in seconds.
 * @returns {Float64Array} Uniformly sampled speed series.
 */
function resampleSpeedProfile(profile, stepSeconds) {
  const { times, speeds } = profile
  const endTime = times[times.length - 1]
  const count = Math.floor(endTime / stepSeconds) + 1
  const out = new Float64Array(count)
  let cursor = 0

  for (let index = 0; index < count; index += 1) {
    const targetTime = index * stepSeconds
    while (cursor + 1 < times.length && times[cursor + 1] < targetTime) {
      cursor += 1
    }
    if (cursor + 1 < times.length && times[cursor + 1] > times[cursor]) {
      const ratio = Math.min(1, Math.max(0, (targetTime - times[cursor]) / (times[cursor + 1] - times[cursor])))
      out[index] = speeds[cursor] + ratio * (speeds[cursor + 1] - speeds[cursor])
    } else {
      out[index] = speeds[cursor]
    }
  }

  return out
}

/**
 * Evaluates Pearson correlation and mean absolute error at a discrete lag where
 * `videoSeries[i]` aligns with `activitySeries[i + lag]`.
 *
 * @param {Float64Array} activitySeries Uniformly sampled activity speeds.
 * @param {Float64Array} videoSeries Uniformly sampled video speeds.
 * @param {number} lag Integer sample offset (`activityIndex - videoIndex`).
 * @param {number} minOverlapSamples Minimum overlapping sample count required.
 * @returns {{r: number, mae: number}|null} Correlation metrics, or null if invalid.
 */
function evaluateSpeedLag(activitySeries, videoSeries, lag, minOverlapSamples) {
  const start = Math.max(0, -lag)
  const end = Math.min(videoSeries.length, activitySeries.length - lag)
  const count = end - start
  if (count < minOverlapSamples) return null

  let sumV = 0
  let sumA = 0
  let sumVV = 0
  let sumAA = 0
  let sumVA = 0
  let sumAbsDiff = 0

  for (let index = start; index < end; index += 1) {
    const v = videoSeries[index]
    const a = activitySeries[index + lag]
    sumV += v
    sumA += a
    sumVV += v * v
    sumAA += a * a
    sumVA += v * a
    sumAbsDiff += Math.abs(v - a)
  }

  const cov = sumVA - (sumV * sumA) / count
  const varV = sumVV - (sumV * sumV) / count
  const varA = sumAA - (sumA * sumA) / count
  if (varV <= 1e-6 || varA <= 1e-6) return null

  return {
    r: cov / Math.sqrt(varV * varA),
    mae: sumAbsDiff / count,
  }
}

/**
 * Aligns embedded video GPS speed telemetry against external activity speed
 * telemetry using two-stage coarse-to-fine cross-correlation.
 *
 * @param {object|null|undefined} videoTelemetry Embedded video telemetry (`stashedVideoTelemetry`).
 * @param {object|null|undefined} externalActivity External activity telemetry (`parsedActivity`).
 * @returns {number|null} Best sync offset in seconds (rounded to 0.1s), or null when unavailable.
 */
export function correlateVideoTelemetryOffset(videoTelemetry, externalActivity) {
  const videoProfile = extractSpeedProfile(videoTelemetry)
  const activityProfile = extractSpeedProfile(externalActivity)
  if (videoProfile === null || activityProfile === null) return null

  const minOverlapSeconds = Math.max(
    CORRELATION_MIN_OVERLAP_FLOOR_SECONDS,
    Math.min(CORRELATION_MIN_OVERLAP_CAP_SECONDS, 0.5 * Math.min(videoProfile.duration, activityProfile.duration)),
  )

  const activityCoarse = resampleSpeedProfile(activityProfile, CORRELATION_COARSE_STEP_SECONDS)
  const videoCoarse = resampleSpeedProfile(videoProfile, CORRELATION_COARSE_STEP_SECONDS)
  const minCoarseSamples = Math.ceil(minOverlapSeconds / CORRELATION_COARSE_STEP_SECONDS)

  const minCoarseLag = -videoCoarse.length + minCoarseSamples
  const maxCoarseLag = activityCoarse.length - minCoarseSamples
  if (minCoarseLag > maxCoarseLag) return null

  let bestCoarseLag = null
  let bestCoarseMetrics = null

  for (let lag = minCoarseLag; lag <= maxCoarseLag; lag += 1) {
    const metrics = evaluateSpeedLag(activityCoarse, videoCoarse, lag, minCoarseSamples)
    if (metrics === null) continue
    if (bestCoarseMetrics === null || metrics.r > bestCoarseMetrics.r) {
      bestCoarseMetrics = metrics
      bestCoarseLag = lag
    }
  }

  if (bestCoarseLag === null || bestCoarseMetrics.r < CORRELATION_MIN_SCORE) return null

  const activityFine = resampleSpeedProfile(activityProfile, CORRELATION_FINE_STEP_SECONDS)
  const videoFine = resampleSpeedProfile(videoProfile, CORRELATION_FINE_STEP_SECONDS)
  const minFineSamples = Math.ceil(minOverlapSeconds / CORRELATION_FINE_STEP_SECONDS)
  const centerFineLag = Math.round((bestCoarseLag * CORRELATION_COARSE_STEP_SECONDS) / CORRELATION_FINE_STEP_SECONDS)
  const fineRadius = Math.round(CORRELATION_FINE_WINDOW_SECONDS / CORRELATION_FINE_STEP_SECONDS)

  let bestFineLag = centerFineLag
  let bestFineMetrics = bestCoarseMetrics

  for (let lag = centerFineLag - fineRadius; lag <= centerFineLag + fineRadius; lag += 1) {
    const metrics = evaluateSpeedLag(activityFine, videoFine, lag, minFineSamples)
    if (metrics === null) continue
    if (metrics.r > bestFineMetrics.r) {
      bestFineMetrics = metrics
      bestFineLag = lag
    }
  }

  if (bestFineMetrics.r < CORRELATION_MIN_SCORE || bestFineMetrics.mae > CORRELATION_MAX_MEAN_ABS_ERROR_MPS) {
    return null
  }

  return Math.round(bestFineLag * CORRELATION_FINE_STEP_SECONDS * 10) / 10
}

/**
 * Reports whether a video interval overlaps the activity interval.
 *
 * @param {object} options Timing inputs.
 * @param {number} options.videoStart Timeline second where the video interval starts.
 * @param {number} options.videoDuration Video interval duration in seconds.
 * @param {number} [options.activityEnd=Infinity] Activity interval end in seconds.
 * @returns {boolean} Whether the half-open intervals overlap.
 */
export function videoOverlapsActivity({ videoStart, videoDuration, activityEnd = Number.POSITIVE_INFINITY }) {
  return videoStart < activityEnd && videoStart + videoDuration > 0
}

class VideoSyncError extends Error {
  constructor(key, message) {
    super(message)
    this.translationKey = key
  }
}

// External timestamps may be missing or unusable. Preserve the existing
// comparison clock: GPS is zoned; ambiguous local camera clock text is unchanged.
function parseSyncTimestamp(timestamp, source, timezone) {
  if (typeof timestamp !== 'string' || timestamp.trim() === '') return null
  try {
    let normalized
    if (source === 'gps') {
      const values = getZonedDateTimeParts(timestamp, timezone)
      normalized = `${values.year}-${values.month}-${values.day}T${values.hour}:${values.minute}:${values.second}`
    } else {
      normalized = timestamp
        .replace(/\.\d+(?=(?:Z|[+-]\d{2}:?\d{2}| UTC)?$)/, '')
        .replace(/(?:Z|[+-]\d{2}:?\d{2}| UTC)$/, '')
        .trim()
        .replace(' ', 'T')
    }
    const parsed = Date.parse(`${normalized}Z`)
    return Number.isFinite(parsed) ? parsed : null
  } catch (error) {
    if (error instanceof RangeError) return null
    throw error
  }
}

/**
 * Captures Apply Timezone: UTC when checked, local when unchecked. An automatic
 * unchecked selection captures the local interpretation.
 * @param {'utc'|'local'|null|undefined} mode Editor selection.
 * @returns {'utc'|'local'} Captured interpretation.
 */
export function captureVideoSyncTimezoneMode(mode) {
  if (mode === null || mode === undefined) return 'local'
  if (mode !== 'local' && mode !== 'utc') throw new Error('Video sync timezone mode must be local or utc')
  return mode
}

/**
 * Calculates a signed baseline independently of clip overlap. GPS and other
 * trusted sources keep absolute-time handling; camera/filename timestamps use
 * the selected interpretation. Interactive auto-sync resolves that selection
 * before batch calibration captures it.
 * @param {object} video Source metadata with creationTime and timeSource.
 * @param {object} activitySummary Activity syncTime, endTime and timezone.
 * @param {'utc'|'local'} timezoneMode Captured Apply Timezone interpretation.
 * @returns {{automaticOffsetSeconds: number, activityDurationSeconds: number}} Raw timing.
 */
function calculateAutomaticVideoOffset(video, activitySummary, timezoneMode) {
  if (!video.creationTime) {
    throw new VideoSyncError('store.couldNotDetermineVideoCreationTime', 'Could not determine video creation time')
  }
  const timezone = activitySummary?.timezone
  if (!timezone) throw new VideoSyncError('store.timezoneIsRequiredForVideoSync', 'timezone is required for video sync')

  const ambiguous = video.timeSource === 'ffprobe' || video.timeSource === 'filename'
  const source = ambiguous && timezoneMode === 'local' ? 'ffprobe' : 'gps'
  const videoStart = parseSyncTimestamp(video.creationTime, source, timezone)
  const activityStart = parseSyncTimestamp(activitySummary.syncTime, 'gps', timezone)
  const activityEnd = parseSyncTimestamp(activitySummary.endTime, 'gps', timezone)
  if (videoStart === null || activityStart === null || activityEnd === null || activityEnd <= activityStart) {
    throw new VideoSyncError('store.invalidTimestampFormats', 'Invalid timestamp formats')
  }
  return {
    automaticOffsetSeconds: (videoStart - activityStart) / 1000,
    activityDurationSeconds: (activityEnd - activityStart) / 1000,
  }
}

/**
 * Resolves interactive sync state using the same baseline as batch calibration.
 * Ambiguous camera timestamps try both interpretations until a mode is selected;
 * prefer local when both overlap and retain an explicit selection.
 * A sole local match remains automatic for subsequent activity changes.
 * Unresolvable external timestamps or clips without positive activity overlap
 * use the warning/zero UI state; overlapping clips retain their signed offset.
 * @param {object} videoState Imported video state.
 * @param {object|null} activitySummary Activity summary.
 * @returns {object} Store sync fields ready to commit.
 */
export function resolveVideoSyncState(videoState, activitySummary) {
  const mode = captureVideoSyncTimezoneMode(videoState.videoSyncTimezoneMode)
  const ambiguous = videoState.importedVideoTimeSource === 'ffprobe' || videoState.importedVideoTimeSource === 'filename'
  const inferMode = ambiguous && (videoState.videoSyncTimezoneMode === null || videoState.videoSyncTimezoneMode === undefined)
  const modes = inferMode ? ['local', 'utc'] : [mode]

  const correlatedOffset =
    activitySummary !== null &&
    activitySummary !== undefined &&
    videoState.importedVideoTimeSource !== 'filename' &&
    videoState.parsedActivitySource === 'activity-file'
      ? correlateVideoTelemetryOffset(videoState.stashedVideoTelemetry, videoState.parsedActivity)
      : null
  const correlatedOverlaps =
    correlatedOffset !== null &&
    videoOverlapsActivity({
      videoStart: correlatedOffset,
      videoDuration: videoState.importedVideoDuration,
      activityEnd: activitySummary?.durationSeconds ?? Number.POSITIVE_INFINITY,
    })

  try {
    const candidates = modes.map((timezoneMode) => {
      const { automaticOffsetSeconds, activityDurationSeconds } = calculateAutomaticVideoOffset(
        { creationTime: videoState.importedVideoCreationTime, timeSource: videoState.importedVideoTimeSource },
        activitySummary,
        timezoneMode,
      )
      return {
        timezoneMode,
        automaticOffsetSeconds,
        overlaps: videoOverlapsActivity({
          videoStart: automaticOffsetSeconds,
          videoDuration: videoState.importedVideoDuration,
          activityEnd: activityDurationSeconds,
        }),
      }
    })
    const matches = candidates.filter(({ overlaps }) => overlaps)
    const selected = matches[0] ?? candidates[0]
    const inferredLocalOnly = inferMode && matches.length === 1 && selected.timezoneMode === 'local'
    const timezoneMode = ambiguous && !inferredLocalOnly ? selected.timezoneMode : null
    if (correlatedOverlaps) {
      return {
        videoSyncOffsetSeconds: correlatedOffset,
        videoSyncWarning: null,
        videoSyncTimezoneMode: timezoneMode,
      }
    }
    return {
      videoSyncOffsetSeconds: selected.overlaps ? selected.automaticOffsetSeconds : 0,
      videoSyncWarning: selected.overlaps ? null : i18next.t('store.videoCouldNotBeSyncedWithActivity', 'Video could not be synced with activity'),
      videoSyncTimezoneMode: timezoneMode,
    }
  } catch (error) {
    if (!(error instanceof VideoSyncError)) throw error
    if (correlatedOverlaps) {
      return {
        videoSyncOffsetSeconds: correlatedOffset,
        videoSyncWarning: null,
        videoSyncTimezoneMode: null,
      }
    }
    return {
      videoSyncOffsetSeconds: 0,
      videoSyncWarning: i18next.t(error.translationKey, error.message),
      videoSyncTimezoneMode: null,
    }
  }
}

/**
 * Captures the shared correction from the committed reference offset. A null
 * activity selects per-video embedded telemetry; a null reference explicitly
 * means zero correction. A present reference must have a resolvable baseline.
 * @param {object} options Calibration inputs from one synchronous snapshot.
 * @param {object|null} options.activitySummary Shared external activity summary, or null.
 * @param {object|null} options.referenceVideo Effective path, creationTime, timeSource and committedOffsetSeconds, or null.
 * @param {'utc'|'local'|null|undefined} options.timezoneMode Apply Timezone selection.
 * @param {boolean} [options.requireReferenceSync=true] Whether an unusable external timestamp blocks calibration.
 * @returns {object} Serializable calibration, matching the Rust BatchCalibration contract.
 */
export function createBatchCalibration({ activitySummary, referenceVideo, timezoneMode, requireReferenceSync = true }) {
  const mode = captureVideoSyncTimezoneMode(timezoneMode)
  if (activitySummary === null) return { mode: 'embeddedActivity' }
  if (referenceVideo === null) return { mode: 'externalActivity', timezoneMode: mode, correctionSeconds: 0, reference: null }
  if (!Number.isFinite(referenceVideo.committedOffsetSeconds)) throw new Error('Committed video sync offset must be a finite number')
  let baseline
  try {
    baseline = calculateAutomaticVideoOffset(referenceVideo, activitySummary, mode)
  } catch (error) {
    if (!requireReferenceSync && error instanceof VideoSyncError)
      return { mode: 'externalActivity', timezoneMode: mode, correctionSeconds: 0, reference: null }
    throw new Error(`Could not calibrate reference video: ${error.message}`)
  }
  const { automaticOffsetSeconds } = baseline
  return {
    mode: 'externalActivity',
    timezoneMode: mode,
    correctionSeconds: referenceVideo.committedOffsetSeconds - automaticOffsetSeconds,
    reference: {
      path: referenceVideo.path,
      creationTime: referenceVideo.creationTime,
      timeSource: referenceVideo.timeSource,
      committedOffsetSeconds: referenceVideo.committedOffsetSeconds,
      automaticOffsetSeconds,
    },
  }
}

/**
 * Evaluates automatic overlap using only the inspected video's timestamp.
 * Reference overrides and shared calibration belong to render submission.
 * Embedded telemetry is prepared later by its owning job; null overlap
 * explicitly means coverage has not yet been inspected.
 * @param {object} video Inspected source metadata with creationTime, timeSource and duration.
 * @param {object|null} activitySummary Shared external activity summary, or null.
 * @param {'utc'|'local'} timezoneMode Captured Apply Timezone interpretation.
 * @param {boolean} [requireSync=true] Whether unusable external timestamps block rendering. Otherwise start at activity time zero.
 * @returns {{timing: object, hasPositiveOverlap: boolean|null}} Job timing and eligibility.
 */
export function resolveBatchVideoTiming(video, activitySummary, timezoneMode, requireSync = true) {
  if (activitySummary === null) return { timing: { mode: 'embeddedActivity' }, hasPositiveOverlap: null }
  let baseline
  try {
    baseline = calculateAutomaticVideoOffset(video, activitySummary, timezoneMode)
  } catch (error) {
    if (requireSync || !(error instanceof VideoSyncError)) throw error
    return { timing: { mode: 'externalActivity', automaticOffsetSeconds: 0 }, hasPositiveOverlap: null }
  }
  const { automaticOffsetSeconds, activityDurationSeconds } = baseline
  return {
    timing: { mode: 'externalActivity', automaticOffsetSeconds },
    hasPositiveOverlap: videoOverlapsActivity({
      videoStart: automaticOffsetSeconds,
      videoDuration: video.duration,
      activityEnd: activityDurationSeconds,
    }),
  }
}

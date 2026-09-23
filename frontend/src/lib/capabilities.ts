import type { InputMode, QueryTask } from '@/types/api'

/**
 * The assistant's advertised capabilities. Each entry is backed by a real
 * recorded demo case (see src/fixtures), so "View example" always opens a
 * genuine analysis rather than a mockup.
 */
export interface Capability {
  id: string
  mode: InputMode
  task: QueryTask
  title: string
  description: string
  exampleQuery: string
  /** Recorded analysis id to preview this capability. */
  demoCaseId?: string
  /** Human description of the required inputs. */
  inputs: string
}

export const CAPABILITIES: Capability[] = [
  {
    id: 'single-optical',
    mode: 'single_optical',
    task: 'land_cover_analysis',
    title: 'Optical scene analysis',
    description:
      'Ask what land cover and materials are present in a single optical or multispectral scene, with per-class area statistics.',
    exampleQuery: 'What land cover types are present in this scene?',
    demoCaseId: 'SAT-2026-000101',
    inputs: '1 optical / multispectral raster',
  },
  {
    id: 'grounding',
    mode: 'single_optical',
    task: 'grounding',
    title: 'Text-guided grounding',
    description:
      'Locate and outline the regions a text query refers to — “where is the water?”, “show the built-up areas”.',
    exampleQuery: 'Where is the water?',
    demoCaseId: 'SAT-2026-000102',
    inputs: '1 optical raster',
  },
  {
    id: 'sar',
    mode: 'single_sar',
    task: 'optical_sar_analysis',
    title: 'SAR backscatter analysis',
    description:
      'Characterise radar backscatter regimes (smooth, diffuse, double-bounce) with speckle-aware statistics.',
    exampleQuery: 'Describe the SAR backscatter in this image.',
    demoCaseId: 'SAT-2026-000104',
    inputs: '1 SAR raster (VV / VH)',
  },
  {
    id: 'fusion',
    mode: 'optical_sar_pair',
    task: 'optical_sar_analysis',
    title: 'Optical + SAR fusion',
    description:
      'Combine co-registered optical and radar evidence into one land-cover decision, flagging where the sensors disagree.',
    exampleQuery: 'Combine the optical and radar images to map land cover.',
    demoCaseId: 'SAT-2026-000105',
    inputs: 'Optical + SAR pair',
  },
  {
    id: 'change',
    mode: 'bitemporal_pair',
    task: 'change_detection',
    title: 'Bi-temporal change detection',
    description:
      'Detect and describe what changed between two dates, with change transitions corroborated across detectors.',
    exampleQuery: 'What changed between the two dates?',
    demoCaseId: 'SAT-2026-000106',
    inputs: '2 co-registered scenes (t1, t2)',
  },
]

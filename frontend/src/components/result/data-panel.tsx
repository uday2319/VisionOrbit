import type { ReactNode } from 'react'
import type { AnalysisData, Artifact } from '@/types/api'
import {
  isChangeData,
  isFusionData,
  isGroundingData,
  isLandCoverData,
  isSarData,
} from '@/types/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { LandCoverPanel } from '@/components/result/panels/landcover-panel'
import { GroundingPanel } from '@/components/result/panels/grounding-panel'
import { SarPanel } from '@/components/result/panels/sar-panel'
import { FusionPanel } from '@/components/result/panels/fusion-panel'
import { ChangePanel } from '@/components/result/panels/change-panel'

/**
 * Dispatches to the right per-task detail panel by narrowing AnalysisData with
 * the structural type guards from types/api.ts. The `task` field cannot
 * discriminate on its own (single-SAR and fusion both report
 * `optical_sar_analysis`), so we key off the payload shape.
 */
export function DataPanel({ data, artifacts }: { data: AnalysisData; artifacts?: Artifact[] }) {
  let title = 'Analysis details'
  let body: ReactNode

  if (isLandCoverData(data)) {
    title = 'Land-cover analysis'
    body = <LandCoverPanel data={data} />
  } else if (isGroundingData(data)) {
    title = 'Grounding & localisation'
    body = <GroundingPanel data={data} />
  } else if (isSarData(data)) {
    title = 'SAR scattering analysis'
    body = <SarPanel data={data} />
  } else if (isFusionData(data)) {
    title = 'Optical + SAR fusion'
    body = <FusionPanel data={data} />
  } else if (isChangeData(data)) {
    title = 'Change detection'
    body = <ChangePanel data={data} artifacts={artifacts} />
  } else {
    body = (
      <p className="text-sm text-muted-foreground">
        No structured detail is available for this analysis.
      </p>
    )
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent>{body}</CardContent>
    </Card>
  )
}

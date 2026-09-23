import type { InputMeta, Modality } from '@/types/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Stat, StatGrid } from '@/components/result/stat'
import { Layers, Radar, HelpCircle, type LucideIcon } from 'lucide-react'
import { formatInt } from '@/lib/format'
import { prettifyKey } from '@/lib/classes'

const MODALITY_META: Record<Modality, { icon: LucideIcon; label: string }> = {
  optical: { icon: Layers, label: 'Optical' },
  sar: { icon: Radar, label: 'SAR' },
  unknown: { icon: HelpCircle, label: 'Unknown' },
}

function crsLabel(meta: InputMeta): string {
  if (meta.crs_epsg) return `EPSG:${meta.crs_epsg}`
  if (meta.crs) return meta.crs
  return '—'
}

function pixelSizeLabel(meta: InputMeta): string {
  if (!meta.pixel_size) return '—'
  const [x, y] = meta.pixel_size
  const ax = Math.abs(x)
  const ay = Math.abs(y)
  return ax === ay ? `${ax} m` : `${ax} × ${ay} m`
}

function InputCard({ meta }: { meta: InputMeta }) {
  const modality = MODALITY_META[meta.modality] ?? MODALITY_META.unknown
  const Icon = modality.icon
  return (
    <div className="space-y-3 rounded-lg border border-border p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Icon className="h-4 w-4 text-primary" aria-hidden="true" />
        <span className="break-all font-mono text-sm font-medium text-foreground">
          {meta.filename}
        </span>
        <Badge variant="secondary">{modality.label}</Badge>
        <Badge variant={meta.georeferenced ? 'success' : 'warning'}>
          {meta.georeferenced ? 'Georeferenced' : 'No georeferencing'}
        </Badge>
        {meta.decimation > 1 && <Badge variant="outline">Decimated ×{meta.decimation}</Badge>}
      </div>

      <StatGrid>
        <Stat
          label="Dimensions"
          value={`${formatInt(meta.width)} × ${formatInt(meta.height)}`}
          hint="px"
        />
        <Stat label="Bands" value={meta.bands} />
        <Stat label="Data type" value={meta.dtype} />
        <Stat label="Driver" value={meta.driver} />
        <Stat label="CRS" value={crsLabel(meta)} />
        <Stat label="Pixel size" value={pixelSizeLabel(meta)} />
        <Stat label="NoData" value={meta.nodata === null ? '—' : meta.nodata} />
        <Stat label="Bands mapped" value={meta.band_roles.length || '—'} />
      </StatGrid>

      {meta.band_roles.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs text-muted-foreground">Band roles:</span>
          {meta.band_roles.map((role, i) => (
            <Badge key={`${role}-${i}`} variant="outline" className="text-[11px]">
              {prettifyKey(role)}
            </Badge>
          ))}
        </div>
      )}
    </div>
  )
}

export function InputsPanel({ inputs }: { inputs: InputMeta[] }) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">
          Inputs
          <span className="ml-2 text-sm font-normal text-muted-foreground">
            {inputs.length} raster{inputs.length === 1 ? '' : 's'}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {inputs.map((meta, i) => (
          <InputCard key={`${meta.filename}-${i}`} meta={meta} />
        ))}
      </CardContent>
    </Card>
  )
}

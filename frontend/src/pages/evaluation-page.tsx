import {
  Activity,
  BarChart3,
  Droplets,
  Leaf,
  Radar,
  ScanSearch,
  ShieldCheck,
  Trees,
} from 'lucide-react'

const analytics = [
  {
    title: 'NDVI',
    description: 'Analyze vegetation health and density.',
    icon: Leaf,
    value: 'Vegetation',
  },
  {
    title: 'NDWI',
    description: 'Detect and analyze surface water.',
    icon: Droplets,
    value: 'Water',
  },
  {
    title: 'NDBI',
    description: 'Identify built-up and urban areas.',
    icon: Trees,
    value: 'Built-up',
  },
  {
    title: 'Change Detection',
    description: 'Compare imagery across two time periods.',
    icon: Activity,
    value: 'Temporal',
  },
  {
    title: 'Optical + SAR',
    description: 'Combine optical and radar observations.',
    icon: Radar,
    value: 'Fusion',
  },
  {
    title: 'Grounding',
    description: 'Locate regions relevant to your query.',
    icon: ScanSearch,
    value: 'Spatial',
  },
]

export function EvaluationPage() {
  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900">
          Analytics
        </h1>

        <p className="mt-1 text-sm text-slate-500">
          Explore the satellite intelligence capabilities available in
          VisionOrbit.
        </p>
      </div>

      {/* Overview cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="vision-card p-5">
          <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-blue-50">
            <BarChart3 className="h-5 w-5 text-blue-600" />
          </div>

          <p className="text-2xl font-bold text-slate-900">6</p>

          <p className="mt-1 text-xs font-medium text-slate-500">
            Analysis capabilities
          </p>
        </div>

        <div className="vision-card p-5">
          <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-green-50">
            <Leaf className="h-5 w-5 text-green-600" />
          </div>

          <p className="text-2xl font-bold text-slate-900">3</p>

          <p className="mt-1 text-xs font-medium text-slate-500">
            Spectral indices
          </p>
        </div>

        <div className="vision-card p-5">
          <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-purple-50">
            <ShieldCheck className="h-5 w-5 text-purple-600" />
          </div>

          <p className="text-2xl font-bold text-slate-900">
            Evidence
          </p>

          <p className="mt-1 text-xs font-medium text-slate-500">
            Evidence-grounded analysis
          </p>
        </div>
      </div>

      {/* Capabilities */}
      <div>
        <div className="mb-3">
          <h2 className="text-sm font-semibold text-slate-900">
            Satellite Analytics
          </h2>

          <p className="mt-1 text-xs text-slate-500">
            Core analysis modules available in the platform.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {analytics.map((item) => {
            const Icon = item.icon

            return (
              <div
                key={item.title}
                className="vision-card group p-5 transition hover:-translate-y-0.5 hover:border-blue-200 hover:shadow-md"
              >
                <div className="flex items-start justify-between">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-50">
                    <Icon className="h-5 w-5 text-blue-600" />
                  </div>

                  <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-medium text-slate-500">
                    {item.value}
                  </span>
                </div>

                <h3 className="mt-4 text-sm font-semibold text-slate-900">
                  {item.title}
                </h3>

                <p className="mt-1 text-xs leading-relaxed text-slate-500">
                  {item.description}
                </p>

                <div className="mt-4 flex items-center gap-1.5 text-[11px] font-medium text-green-600">
                  <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                  Available
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Workflow */}
      <div className="vision-card p-5">
        <h2 className="text-sm font-semibold text-slate-900">
          Analysis Workflow
        </h2>

        <p className="mt-1 text-xs text-slate-500">
          VisionOrbit converts satellite imagery and natural-language
          questions into evidence-based analysis.
        </p>

        <div className="mt-5 grid grid-cols-1 gap-3 md:grid-cols-4">
          {[
            ['01', 'Upload', 'Provide satellite imagery'],
            ['02', 'Ask', 'Describe what you want to analyze'],
            ['03', 'Analyze', 'Run the appropriate analysis pipeline'],
            ['04', 'Evidence', 'Return results with supporting evidence'],
          ].map(([number, title, description]) => (
            <div
              key={number}
              className="rounded-xl border border-slate-100 bg-slate-50 p-4"
            >
              <span className="text-xs font-bold text-blue-600">
                {number}
              </span>

              <h3 className="mt-2 text-sm font-semibold text-slate-800">
                {title}
              </h3>

              <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
                {description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
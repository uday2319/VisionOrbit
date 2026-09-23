import { useId, useRef, useState } from 'react'
import { UploadCloud, FileImage, X, AlertCircle } from 'lucide-react'
import type { InputSlot } from '@/lib/uploads'
import { ACCEPT_ATTR, formatBytes } from '@/lib/uploads'
import { cn } from '@/lib/utils'

interface FileDropzoneProps {
  slot: InputSlot
  file: File | null
  error: string | null
  disabled?: boolean
  onSelect: (file: File | null) => void
}

export function FileDropzone({ slot, file, error, disabled, onSelect }: FileDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragActive, setDragActive] = useState(false)
  const errorId = useId()

  function pickFrom(list: FileList | null) {
    if (!list || list.length === 0) return
    onSelect(list[0])
  }

  function onDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setDragActive(false)
    if (disabled) return
    pickFrom(event.dataTransfer.files)
  }

  return (
    <div className="space-y-1.5">
      <div
        onDragOver={(e) => {
          e.preventDefault()
          if (!disabled) setDragActive(true)
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={onDrop}
        className={cn(
          'rounded-lg border border-dashed p-4 transition-colors',
          dragActive ? 'border-primary bg-accent' : 'border-input',
          error && 'border-destructive',
          disabled && 'opacity-60',
        )}
      >
        {file ? (
          <div className="flex items-center gap-3">
            <FileImage className="h-8 w-8 shrink-0 text-primary" aria-hidden="true" />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-foreground" title={file.name}>
                {file.name}
              </p>
              <p className="text-xs text-muted-foreground">
                {slot.label} · {formatBytes(file.size)}
              </p>
            </div>
            <button
              type="button"
              onClick={() => {
                onSelect(null)
                if (inputRef.current) inputRef.current.value = ''
              }}
              disabled={disabled}
              className="rounded-md p-1 text-muted-foreground hover:bg-accent hover:text-foreground disabled:opacity-50"
              aria-label={`Remove ${slot.label}`}
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            disabled={disabled}
            className="flex w-full items-center gap-3 text-left disabled:cursor-not-allowed"
          >
            <UploadCloud className="h-8 w-8 shrink-0 text-muted-foreground" aria-hidden="true" />
            <span className="min-w-0">
              <span className="block text-sm font-medium text-foreground">{slot.label}</span>
              <span className="block text-xs text-muted-foreground">{slot.hint}</span>
              <span className="mt-0.5 block text-xs text-primary">
                Drag &amp; drop or click to browse
              </span>
            </span>
          </button>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_ATTR}
          className="sr-only"
          aria-label={slot.label}
          aria-invalid={error ? true : undefined}
          aria-describedby={error ? errorId : undefined}
          disabled={disabled}
          onChange={(e) => {
            const selectedFile = e.target.files?.[0] ?? null
            if (selectedFile) onSelect(selectedFile)
          }}
        />
      </div>
      {error && (
        <p id={errorId} className="flex items-center gap-1.5 text-xs text-destructive">
          <AlertCircle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          {error}
        </p>
      )}
    </div>
  )
}

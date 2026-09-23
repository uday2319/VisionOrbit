import { useState, useRef, useEffect, useCallback } from 'react'
import { GripVertical } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ComparisonSliderProps {
  beforeSrc: string
  afterSrc: string
  beforeLabel?: string
  afterLabel?: string
  className?: string
}

export function ComparisonSlider({
  beforeSrc,
  afterSrc,
  beforeLabel = 'Before',
  afterLabel = 'After',
  className,
}: ComparisonSliderProps) {
  const [sliderPosition, setSliderPosition] = useState(50)
  const [isDragging, setIsDragging] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const handleMove = useCallback(
    (clientX: number) => {
      if (!containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const x = Math.max(0, Math.min(clientX - rect.left, rect.width))
      const percent = Math.max(0, Math.min((x / rect.width) * 100, 100))
      setSliderPosition(percent)
    },
    []
  )

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging) return
      e.preventDefault()
      handleMove(e.clientX)
    },
    [isDragging, handleMove]
  )

  const handleTouchMove = useCallback(
    (e: TouchEvent) => {
      if (!isDragging) return
      handleMove(e.touches[0].clientX)
    },
    [isDragging, handleMove]
  )

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove)
      window.addEventListener('mouseup', handleMouseUp)
      window.addEventListener('touchmove', handleTouchMove, { passive: false })
      window.addEventListener('touchend', handleMouseUp)
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
      window.removeEventListener('touchmove', handleTouchMove)
      window.removeEventListener('touchend', handleMouseUp)
    }
  }, [isDragging, handleMouseMove, handleTouchMove, handleMouseUp])

  return (
    <div
      ref={containerRef}
      className={cn(
        'relative w-full overflow-hidden rounded-lg select-none',
        className
      )}
      style={{ aspectRatio: '1 / 1' }}
    >
      {/* After image (background) */}
      <img
        src={afterSrc}
        alt={afterLabel}
        className="absolute inset-0 h-full w-full object-cover"
        draggable={false}
      />
      
      {afterLabel && (
        <div className="absolute right-4 top-4 rounded-md bg-background/70 px-2 py-1 text-xs font-medium text-foreground backdrop-blur-sm z-10">
          {afterLabel}
        </div>
      )}

      {/* Before image (foreground) */}
      <img
        src={beforeSrc}
        alt={beforeLabel}
        className="absolute inset-0 h-full w-full object-cover"
        style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
        draggable={false}
      />
      
      {beforeLabel && sliderPosition > 10 && (
        <div className="absolute left-4 top-4 rounded-md bg-background/70 px-2 py-1 text-xs font-medium text-foreground backdrop-blur-sm z-10">
          {beforeLabel}
        </div>
      )}

      {/* Slider handle */}
      <div
        className="absolute inset-y-0 w-1 cursor-ew-resize bg-primary/80 transition-colors hover:bg-primary z-20"
        style={{ left: `calc(${sliderPosition}% - 2px)` }}
        onMouseDown={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onTouchStart={() => {
          setIsDragging(true)
        }}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex h-8 w-8 items-center justify-center rounded-full bg-background border border-border shadow-md transition-transform duration-200 ease-in-out hover:scale-110">
          <GripVertical className="h-4 w-4 text-foreground" />
        </div>
      </div>
    </div>
  )
}

import { RouterProvider } from 'react-router-dom'
import { ThemeProvider } from '@/components/theme/theme-provider'
import { TooltipProvider } from '@/components/ui/tooltip'
import { router } from '@/router'

export default function App() {
  return (
    <ThemeProvider>
      <TooltipProvider delayDuration={200}>
        <RouterProvider router={router} />
      </TooltipProvider>
    </ThemeProvider>
  )
}

import * as React from 'react'
const NextThemesProvider = dynamic(
  () => import('next-themes').then((e) => e.ThemeProvider),
  {
    ssr: false,
  }
)

import { ThemeProvider as NextThemesProviderX } from "next-themes";
type ThemeProviderProps = React.ComponentProps<typeof NextThemesProviderX>;

import dynamic from 'next/dynamic'

export function ThemeProvider({ children, ...props }: ThemeProviderProps) {
  return <NextThemesProvider {...props}>{children}</NextThemesProvider>
}
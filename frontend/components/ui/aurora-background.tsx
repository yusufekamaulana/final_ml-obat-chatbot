"use client";
import { cn } from "@/lib/utils";
import React, { ReactNode } from "react";

interface AuroraBackgroundProps extends React.HTMLProps<HTMLDivElement> {
  children: ReactNode;
  showRadialGradient?: boolean;
}

export const AuroraBackground = ({
  className,
  children,
  showRadialGradient = true,
  ...props
}: AuroraBackgroundProps) => {
  return (
    <main>
      <div
        className={cn(
          "transition-bg relative flex min-h-screen flex-col items-center justify-center text-slate-950",
          className
        )}
        {...props}
      >
        <div
          className="absolute inset-0 overflow-hidden"
          style={{
            "--aurora":
              "repeating-linear-gradient(100deg,#a7f3d0_10%,#5eead4_15%,#bfdbfe_20%,#93c5fd_25%,#e0f2fe_30%)",
            "--dark-gradient":
              "repeating-linear-gradient(100deg,#0f172a_0%,#0f172a_7%,transparent_10%,transparent_12%,#0f172a_16%)",
            "--white-gradient":
              "repeating-linear-gradient(100deg,#fff_0%,#fff_7%,transparent_10%,transparent_12%,#fff_16%)",

            "--green-200": "#a7f3d0",
            "--teal-300": "#5eead4",
            "--blue-200": "#bfdbfe",
            "--blue-300": "#93c5fd",
            "--blue-100": "#e0f2fe",
            "--black": "#0f172a",
            "--white": "#fff",
            "--transparent": "transparent",
          } as React.CSSProperties}
        >
          <div
            className={cn(
              `after:animate-aurora pointer-events-none absolute -inset-[10px]
              [background-image:var(--white-gradient),var(--aurora)]
              [background-size:300%,_200%]
              [background-position:50%_50%,50%_50%]
              opacity-15 blur-sm filter will-change-transform
              [--aurora:repeating-linear-gradient(100deg,var(--green-200)_10%,var(--teal-300)_15%,var(--blue-200)_20%,var(--blue-300)_25%,var(--blue-100)_30%)]
              [--dark-gradient:repeating-linear-gradient(100deg,var(--black)_0%,var(--black)_7%,var(--transparent)_10%,var(--transparent)_12%,var(--black)_16%)]
              [--white-gradient:repeating-linear-gradient(100deg,var(--white)_0%,var(--white)_7%,var(--transparent)_10%,var(--transparent)_12%,var(--white)_16%)]
              after:absolute after:inset-0 after:[background-image:var(--white-gradient),var(--aurora)]
              after:[background-size:200%,_100%]
              after:[background-attachment:fixed]
              after:mix-blend-overlay
              after:content-[""]
              dark:[background-image:var(--dark-gradient),var(--aurora)]`,

              showRadialGradient &&
                `[mask-image:radial-gradient(ellipse_at_100%_0%,black_10%,var(--transparent)_70%)]`
            )}
          ></div>
        </div>

        {children}
      </div>
    </main>
  );
};

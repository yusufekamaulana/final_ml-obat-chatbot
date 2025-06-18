import { ArrowRightIcon } from "@radix-ui/react-icons";
import { ComponentPropsWithoutRef, ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface BentoGridProps extends ComponentPropsWithoutRef<"div"> {
  children: ReactNode;
  className?: string;
}

interface BentoCardProps extends ComponentPropsWithoutRef<"div"> {
  name: string;
  className?: string;
  background: ReactNode;
  Icon: React.ElementType;
  description: string;
  href: string;
  cta: string;
}

const BentoGrid = ({ children, className, ...props }: BentoGridProps) => {
  return (
    <div
      className={cn(
        "grid w-full grid-cols-1 sm:grid-cols-2 gap-4", // Responsif: 1 kolom di mobile, 2 di tablet+
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};

const BentoCard = ({
  name,
  className,
  background,
  Icon,
  description,
  href,
  cta,
  ...props
}: BentoCardProps) => (
  <div
    key={name}
    className={cn(
      "group relative flex flex-col justify-between overflow-hidden rounded-xl",
      // Ukuran kotak persegi
      "aspect-square",
      // Gaya latar belakang dan bayangan
      "bg-background shadow-sm dark:bg-background dark:border dark:border-white/[.1]",
      className
    )}
    {...props}
  >
    <div>{background}</div>

    <div className="pointer-events-none z-10 flex flex-col gap-1 p-6 transition-all duration-300 group-hover:-translate-y-6">
      <Icon className="h-12 w-12 text-neutral-700 transition-transform group-hover:scale-90 dark:text-neutral-200" />
      <h3 className="text-xl font-semibold text-neutral-700 dark:text-neutral-200">
        {name}
      </h3>
      <p className="text-sm text-neutral-500 dark:text-neutral-400">{description}</p>
    </div>

    <div className="pointer-events-none absolute bottom-0 w-full translate-y-10 opacity-0 transition-all duration-300 group-hover:translate-y-0 group-hover:opacity-100 p-4">
      <Button variant="ghost" asChild size="sm" className="pointer-events-auto">
        <a href={href}>
          {cta}
          <ArrowRightIcon className="ms-2 h-4 w-4 rtl:rotate-180" />
        </a>
      </Button>
    </div>

    <div className="pointer-events-none absolute inset-0 transition-all duration-300 group-hover:bg-black/5 dark:group-hover:bg-white/5" />
  </div>
);

export { BentoGrid, BentoCard };

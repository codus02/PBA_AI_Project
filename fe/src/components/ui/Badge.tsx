import { cn } from '@/lib/utils';
import { HTMLAttributes } from 'react';

type BadgeVariant = 'default' | 'amber' | 'green' | 'blue' | 'red' | 'purple';

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

export default function Badge({ variant = 'default', className, children, ...props }: BadgeProps) {
  const variants: Record<BadgeVariant, string> = {
    default: 'bg-zinc-800 text-zinc-300 border-zinc-700',
    amber: 'bg-amber-400/15 text-amber-300 border-amber-400/30',
    green: 'bg-emerald-400/15 text-emerald-300 border-emerald-400/30',
    blue: 'bg-blue-400/15 text-blue-300 border-blue-400/30',
    red: 'bg-red-400/15 text-red-300 border-red-400/30',
    purple: 'bg-purple-400/15 text-purple-300 border-purple-400/30',
  };

  return (
    <span
      className={cn(
        'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border',
        variants[variant],
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}

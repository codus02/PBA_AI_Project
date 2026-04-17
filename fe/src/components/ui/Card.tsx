import { cn } from '@/lib/utils';
import { HTMLAttributes } from 'react';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  glow?: boolean;
}

export default function Card({ glow = false, className, children, ...props }: CardProps) {
  return (
    <div
      className={cn(
        'bg-zinc-900 border border-zinc-800 rounded-2xl p-5',
        glow && 'shadow-lg shadow-amber-400/5 border-amber-400/20',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({ className, children, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('mb-4', className)} {...props}>
      {children}
    </div>
  );
}

export function CardTitle({ className, children, ...props }: HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3 className={cn('text-lg font-semibold text-zinc-100', className)} {...props}>
      {children}
    </h3>
  );
}

export function CardBody({ className, children, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('text-zinc-300', className)} {...props}>
      {children}
    </div>
  );
}

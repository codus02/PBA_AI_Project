import { cn } from '@/lib/utils';

interface Step {
  label: string;
  icon: string;
}

interface StepIndicatorProps {
  steps: Step[];
  current: number;
}

export default function StepIndicator({ steps, current }: StepIndicatorProps) {
  return (
    <div className="flex items-center gap-0">
      {steps.map((step, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <div key={i} className="flex items-center">
            <div className="flex flex-col items-center gap-1">
              <div
                className={cn(
                  'w-8 h-8 rounded-full flex items-center justify-center text-sm transition-all',
                  done && 'bg-amber-400 text-zinc-950',
                  active && 'bg-amber-400/20 text-amber-400 border-2 border-amber-400',
                  !done && !active && 'bg-zinc-800 text-zinc-500 border border-zinc-700'
                )}
              >
                {done ? '✓' : step.icon}
              </div>
              <span
                className={cn(
                  'text-xs',
                  active ? 'text-amber-400' : done ? 'text-zinc-400' : 'text-zinc-600'
                )}
              >
                {step.label}
              </span>
            </div>
            {i < steps.length - 1 && (
              <div
                className={cn(
                  'h-px w-8 mx-1 mb-5 transition-all',
                  done ? 'bg-amber-400' : 'bg-zinc-700'
                )}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

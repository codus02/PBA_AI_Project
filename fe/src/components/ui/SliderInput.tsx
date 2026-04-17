'use client';

import { cn } from '@/lib/utils';

interface SliderInputProps {
  label: string;
  value: number | null;
  min?: number;
  max?: number;
  onChange: (value: number) => void;
  leftLabel?: string;
  rightLabel?: string;
}

export default function SliderInput({
  label,
  value,
  min = 1,
  max = 5,
  onChange,
  leftLabel,
  rightLabel,
}: SliderInputProps) {
  const steps = Array.from({ length: max - min + 1 }, (_, i) => i + min);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-zinc-300">{label}</span>
        <span className="text-sm text-amber-400 font-semibold">
          {value !== null ? value : <span className="text-zinc-600">미선택</span>}
        </span>
      </div>
      <div className="flex items-center gap-2">
        {steps.map((step) => (
          <button
            key={step}
            type="button"
            onClick={() => onChange(step)}
            className={cn(
              'flex-1 h-8 rounded-lg text-xs font-semibold transition-all duration-150',
              value === step
                ? 'bg-amber-400 text-zinc-950'
                : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700 border border-zinc-700'
            )}
          >
            {step}
          </button>
        ))}
      </div>
      {(leftLabel || rightLabel) && (
        <div className="flex justify-between text-xs text-zinc-500">
          <span>{leftLabel}</span>
          <span>{rightLabel}</span>
        </div>
      )}
    </div>
  );
}

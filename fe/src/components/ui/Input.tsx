import { cn } from '@/lib/utils';
import { InputHTMLAttributes, TextareaHTMLAttributes, forwardRef } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, className, ...props }, ref) => (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label className="text-sm font-medium text-zinc-300">{label}</label>
      )}
      <input
        ref={ref}
        className={cn(
          'h-10 px-3 rounded-xl bg-zinc-800 border border-zinc-700 text-zinc-100 placeholder:text-zinc-500',
          'focus:outline-none focus:ring-2 focus:ring-amber-400/50 focus:border-amber-400/50',
          'transition-all duration-150',
          error && 'border-red-500/50',
          className
        )}
        {...props}
      />
      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  )
);
Input.displayName = 'Input';

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ label, error, className, ...props }, ref) => (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label className="text-sm font-medium text-zinc-300">{label}</label>
      )}
      <textarea
        ref={ref}
        className={cn(
          'px-3 py-2.5 rounded-xl bg-zinc-800 border border-zinc-700 text-zinc-100 placeholder:text-zinc-500 resize-none',
          'focus:outline-none focus:ring-2 focus:ring-amber-400/50 focus:border-amber-400/50',
          'transition-all duration-150',
          error && 'border-red-500/50',
          className
        )}
        {...props}
      />
      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  )
);
Textarea.displayName = 'Textarea';

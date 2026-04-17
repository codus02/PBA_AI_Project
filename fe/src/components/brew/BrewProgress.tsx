'use client';

import type { BrewStatus } from '@/lib/types';
import { cn } from '@/lib/utils';

const STEPS: { status: BrewStatus; label: string; emoji: string; desc: string }[] = [
  { status: 'preparing', label: '준비 중', emoji: '🧊', desc: '얼음과 글라스를 준비하는 중입니다' },
  { status: 'pouring', label: '주입 중', emoji: '🍾', desc: '재료를 정확한 비율로 주입하는 중입니다' },
  { status: 'mixing', label: '혼합 중', emoji: '🌀', desc: '재료를 혼합하는 중입니다' },
  { status: 'complete', label: '완료', emoji: '🍹', desc: '칵테일이 완성되었습니다!' },
];

const STATUS_ORDER: BrewStatus[] = ['idle', 'preparing', 'pouring', 'mixing', 'complete'];

interface BrewProgressProps {
  status: BrewStatus;
  progress: number;
}

export default function BrewProgress({ status, progress }: BrewProgressProps) {
  const currentIndex = STATUS_ORDER.indexOf(status);

  return (
    <div className="flex flex-col gap-6">
      {/* 전체 진행 바 */}
      <div>
        <div className="flex justify-between text-xs text-zinc-400 mb-2">
          <span>제조 진행률</span>
          <span>{progress}%</span>
        </div>
        <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-amber-400 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* 단계 표시 */}
      <div className="flex flex-col gap-3">
        {STEPS.map((step, i) => {
          const stepIndex = STATUS_ORDER.indexOf(step.status);
          const isDone = currentIndex > stepIndex;
          const isActive = currentIndex === stepIndex;

          return (
            <div
              key={step.status}
              className={cn(
                'flex items-center gap-4 p-4 rounded-xl border transition-all duration-300',
                isDone && 'bg-emerald-400/10 border-emerald-400/20',
                isActive && 'bg-amber-400/10 border-amber-400/30',
                !isDone && !isActive && 'bg-zinc-900 border-zinc-800 opacity-40'
              )}
            >
              <div
                className={cn(
                  'w-10 h-10 rounded-full flex items-center justify-center text-xl',
                  isDone && 'bg-emerald-400/20',
                  isActive && 'bg-amber-400/20',
                  !isDone && !isActive && 'bg-zinc-800'
                )}
              >
                {isDone ? '✓' : (
                  isActive ? (
                    <span className="animate-pulse">{step.emoji}</span>
                  ) : (
                    step.emoji
                  )
                )}
              </div>
              <div className="flex-1">
                <p className={cn(
                  'font-medium text-sm',
                  isDone && 'text-emerald-400',
                  isActive && 'text-amber-400',
                  !isDone && !isActive && 'text-zinc-500'
                )}>
                  {step.label}
                </p>
                {isActive && (
                  <p className="text-xs text-zinc-400 mt-0.5">{step.desc}</p>
                )}
              </div>
              {isActive && (
                <div className="w-4 h-4 border-2 border-amber-400 border-t-transparent rounded-full animate-spin" />
              )}
            </div>
          );
        })}
      </div>

      {status === 'complete' && (
        <div className="text-center py-4">
          <div className="text-6xl mb-3 animate-bounce">🍹</div>
          <p className="text-xl font-bold text-amber-400">칵테일 완성!</p>
          <p className="text-sm text-zinc-400 mt-1">즐거운 파티 되세요!</p>
        </div>
      )}
    </div>
  );
}

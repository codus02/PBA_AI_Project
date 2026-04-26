'use client';

import { useState, useEffect } from 'react';
import type { BrewStatus } from '@/lib/types';
import { cn } from '@/lib/utils';

const STEPS: { status: BrewStatus; emoji: string; desc: string; doneLabel: string }[] = [
  { status: 'preparing', emoji: '🧊', desc: '얼음과 글라스를 준비하는 중입니다', doneLabel: '준비 완료' },
  { status: 'pouring', emoji: '🍾', desc: '재료를 정확한 비율로 주입하는 중입니다', doneLabel: '주입 완료' },
  { status: 'mixing', emoji: '🌀', desc: '재료를 혼합하는 중입니다', doneLabel: '혼합 완료' },
  { status: 'complete', emoji: '🍹', desc: '칵테일이 완성되었습니다!', doneLabel: '완성' },
];

const STATUS_ORDER: BrewStatus[] = ['idle', 'preparing', 'pouring', 'mixing', 'complete'];
const FRAME_COUNT = 8;
const FRAME_MS = 150;

interface BrewProgressProps {
  status: BrewStatus;
  progress: number;
}

export default function BrewProgress({ status, progress }: BrewProgressProps) {
  const currentIndex = STATUS_ORDER.indexOf(status);
  const [frame, setFrame] = useState(1);

  useEffect(() => {
    if (status === 'idle') return;
    const id = setInterval(() => setFrame((f) => (f % FRAME_COUNT) + 1), FRAME_MS);
    return () => clearInterval(id);
  }, [status]);

  return (
    <div className="flex flex-col gap-5">
      {/* 바텐더 애니메이션 + 진행 바 */}
      <div>
        <div className="flex justify-center mb-3">
          <img
            src={`/image_${frame}.png`}
            alt="바텐더"
            className="h-36 object-contain"
          />
        </div>
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
      <div className="flex flex-col gap-2">
        {STEPS.map((step) => {
          const stepIndex = STATUS_ORDER.indexOf(step.status);
          const isDone =
            currentIndex > stepIndex ||
            (status === 'complete' && step.status === 'complete');
          const isActive = !isDone && currentIndex === stepIndex;

          return (
            <div
              key={step.status}
              className={cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-xl border transition-all duration-300',
                isDone && 'bg-emerald-400/10 border-emerald-400/20',
                isActive && 'bg-amber-400/10 border-amber-400/30',
                !isDone && !isActive && 'bg-zinc-900 border-zinc-800 opacity-40'
              )}
            >
              <div
                className={cn(
                  'w-8 h-8 rounded-full flex items-center justify-center text-base shrink-0',
                  isDone && 'bg-emerald-400/20',
                  isActive && 'bg-amber-400/20',
                  !isDone && !isActive && 'bg-zinc-800'
                )}
              >
                {isDone ? (
                  <span className="text-emerald-400 font-bold text-sm">✓</span>
                ) : isActive ? (
                  <span className="animate-pulse">{step.emoji}</span>
                ) : (
                  step.emoji
                )}
              </div>
              <p
                className={cn(
                  'text-sm flex-1',
                  isDone && 'text-emerald-400 font-medium',
                  isActive && 'text-amber-300',
                  !isDone && !isActive && 'text-zinc-500'
                )}
              >
                {isDone ? step.doneLabel : step.desc}
              </p>
              {isActive && (
                <div className="w-3.5 h-3.5 border-2 border-amber-400 border-t-transparent rounded-full animate-spin shrink-0" />
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

'use client';

import { useState } from 'react';
import type { GuestPreferences, AromaType } from '@/lib/types';
import Button from '@/components/ui/Button';
import Card, { CardHeader, CardTitle, CardBody } from '@/components/ui/Card';
import SliderInput from '@/components/ui/SliderInput';
import { cn } from '@/lib/utils';

interface TasteFormProps {
  onSubmit: (prefs: GuestPreferences) => void;
}

type FormState = {
  sweetness: number | null;
  sourness: number | null;
  bitterness: number | null;
  spiciness: number | null;
  aromas: AromaType[];
  alcoholTolerance: GuestPreferences['alcoholTolerance'] | null;
  intensity: GuestPreferences['intensity'] | null;
  experience: GuestPreferences['experience'] | null;
};

const AROMAS: { id: AromaType; label: string; emoji: string }[] = [
  { id: 'citrus', label: '시트러스', emoji: '🍋' },
  { id: 'floral', label: '꽃향기', emoji: '🌸' },
  { id: 'woody', label: '우디', emoji: '🪵' },
  { id: 'herbal', label: '허브', emoji: '🌿' },
  { id: 'fruity', label: '과일', emoji: '🍓' },
  { id: 'spicy', label: '스파이시', emoji: '🌶️' },
  { id: 'smoky', label: '스모키', emoji: '🔥' },
  { id: 'sweet', label: '달콤', emoji: '🍯' },
];

export default function TasteForm({ onSubmit }: TasteFormProps) {
  const [prefs, setPrefs] = useState<FormState>({
    sweetness: null,
    sourness: null,
    bitterness: null,
    spiciness: null,
    aromas: [],
    alcoholTolerance: null,
    intensity: null,
    experience: null,
  });

  const toggleAroma = (aroma: AromaType) => {
    setPrefs((p) => ({
      ...p,
      aromas: p.aromas.includes(aroma)
        ? p.aromas.filter((a) => a !== aroma)
        : [...p.aromas, aroma],
    }));
  };

  const isComplete =
    prefs.sweetness !== null &&
    prefs.sourness !== null &&
    prefs.bitterness !== null &&
    prefs.spiciness !== null &&
    prefs.alcoholTolerance !== null &&
    prefs.intensity !== null &&
    prefs.experience !== null;

  const handleSubmit = () => {
    if (!isComplete) return;
    onSubmit({
      sweetness: prefs.sweetness!,
      sourness: prefs.sourness!,
      bitterness: prefs.bitterness!,
      spiciness: prefs.spiciness!,
      aromas: prefs.aromas,
      alcoholTolerance: prefs.alcoholTolerance!,
      intensity: prefs.intensity!,
      experience: prefs.experience!,
    });
  };

  return (
    <div className="flex flex-col gap-5">
      <Card>
        <CardHeader>
          <CardTitle>맛 선호도</CardTitle>
        </CardHeader>
        <CardBody className="flex flex-col gap-4">
          <SliderInput
            label="단맛"
            value={prefs.sweetness}
            onChange={(v) => setPrefs((p) => ({ ...p, sweetness: v }))}
            leftLabel="싫어요"
            rightLabel="좋아요"
          />
          <SliderInput
            label="신맛"
            value={prefs.sourness}
            onChange={(v) => setPrefs((p) => ({ ...p, sourness: v }))}
            leftLabel="싫어요"
            rightLabel="좋아요"
          />
          <SliderInput
            label="쓴맛"
            value={prefs.bitterness}
            onChange={(v) => setPrefs((p) => ({ ...p, bitterness: v }))}
            leftLabel="싫어요"
            rightLabel="좋아요"
          />
          <SliderInput
            label="매운맛"
            value={prefs.spiciness}
            onChange={(v) => setPrefs((p) => ({ ...p, spiciness: v }))}
            leftLabel="싫어요"
            rightLabel="좋아요"
          />
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>향 선호도 <span className="text-sm text-zinc-500 font-normal">(복수 선택)</span></CardTitle>
        </CardHeader>
        <CardBody>
          <div className="grid grid-cols-4 gap-2">
            {AROMAS.map((a) => (
              <button
                key={a.id}
                type="button"
                onClick={() => toggleAroma(a.id)}
                className={cn(
                  'flex flex-col items-center gap-1 p-2 rounded-xl text-xs transition-all border',
                  prefs.aromas.includes(a.id)
                    ? 'bg-amber-400/20 border-amber-400/50 text-amber-300'
                    : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:bg-zinc-700'
                )}
              >
                <span className="text-xl">{a.emoji}</span>
                <span>{a.label}</span>
              </button>
            ))}
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>알코올 & 강도</CardTitle>
        </CardHeader>
        <CardBody className="flex flex-col gap-4">
          <div>
            <p className="text-sm font-medium text-zinc-300 mb-2">알코올 내성</p>
            <div className="grid grid-cols-4 gap-2">
              {(['none', 'low', 'medium', 'high'] as const).map((v) => {
                const labels = { none: '없음', low: '약하게', medium: '보통', high: '강하게' };
                return (
                  <button
                    key={v}
                    type="button"
                    onClick={() => setPrefs((p) => ({ ...p, alcoholTolerance: v }))}
                    className={cn(
                      'h-9 rounded-xl text-sm font-medium transition-all border',
                      prefs.alcoholTolerance === v
                        ? 'bg-amber-400/20 border-amber-400/50 text-amber-300'
                        : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:bg-zinc-700'
                    )}
                  >
                    {labels[v]}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <p className="text-sm font-medium text-zinc-300 mb-2">원하는 강도감</p>
            <div className="grid grid-cols-3 gap-2">
              {(['light', 'medium', 'strong'] as const).map((v) => {
                const labels = { light: '가볍게', medium: '보통', strong: '강하게' };
                return (
                  <button
                    key={v}
                    type="button"
                    onClick={() => setPrefs((p) => ({ ...p, intensity: v }))}
                    className={cn(
                      'h-9 rounded-xl text-sm font-medium transition-all border',
                      prefs.intensity === v
                        ? 'bg-amber-400/20 border-amber-400/50 text-amber-300'
                        : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:bg-zinc-700'
                    )}
                  >
                    {labels[v]}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <p className="text-sm font-medium text-zinc-300 mb-2">칵테일 경험</p>
            <div className="grid grid-cols-3 gap-2">
              {(['beginner', 'casual', 'experienced'] as const).map((v) => {
                const labels = { beginner: '거의 없음', casual: '가끔', experienced: '자주' };
                return (
                  <button
                    key={v}
                    type="button"
                    onClick={() => setPrefs((p) => ({ ...p, experience: v }))}
                    className={cn(
                      'h-9 rounded-xl text-sm font-medium transition-all border',
                      prefs.experience === v
                        ? 'bg-amber-400/20 border-amber-400/50 text-amber-300'
                        : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:bg-zinc-700'
                    )}
                  >
                    {labels[v]}
                  </button>
                );
              })}
            </div>
          </div>
        </CardBody>
      </Card>

      <Button size="lg" className="w-full" disabled={!isComplete} onClick={handleSubmit}>
        다음 단계로 →
      </Button>
    </div>
  );
}

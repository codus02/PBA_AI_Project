'use client';

import { useState } from 'react';
import type { GuestPreferences, TasteTag, AromaTag } from '@/lib/types';
import Button from '@/components/ui/Button';
import Card, { CardHeader, CardTitle, CardBody } from '@/components/ui/Card';
import { cn } from '@/lib/utils';

interface TasteFormProps {
  onSubmit: (prefs: GuestPreferences) => void;
}

type FormState = {
  experience: GuestPreferences['experience'] | null;
  alcoholTolerance: GuestPreferences['alcoholTolerance'] | null;
  tasteTags: TasteTag[];
  aromaTags: AromaTag[];
};

const TASTE_TAGS: { id: TasteTag; label: string }[] = [
  { id: 'sweet', label: '단맛' },
  { id: 'sour', label: '신맛' },
  { id: 'bitter', label: '쓴맛' },
  { id: 'refreshing', label: '청량함' },
  { id: 'body', label: '바디감' },
  { id: 'creamy', label: '크리미함' },
];

const AROMA_TAGS: { id: AromaTag; label: string }[] = [
  { id: 'fruity', label: '과일향' },
  { id: 'herbal', label: '허브향' },
  { id: 'mint', label: '민트향' },
  { id: 'citrus', label: '시트러스향' },
  { id: 'woody', label: '우디향' },
  { id: 'coffee', label: '커피향' },
  { id: 'floral', label: '꽃향' },
];

export default function TasteForm({ onSubmit }: TasteFormProps) {
  const [prefs, setPrefs] = useState<FormState>({
    experience: null,
    alcoholTolerance: null,
    tasteTags: [],
    aromaTags: [],
  });

  const toggleTag = <T extends string>(arr: T[], val: T): T[] =>
    arr.includes(val) ? arr.filter((v) => v !== val) : [...arr, val];

  const isComplete = prefs.experience !== null && prefs.alcoholTolerance !== null;

  const handleSubmit = () => {
    if (!isComplete) return;
    onSubmit({
      experience: prefs.experience!,
      alcoholTolerance: prefs.alcoholTolerance!,
      tasteTags: prefs.tasteTags,
      aromaTags: prefs.aromaTags,
    });
  };

  const tagButtonClass = (selected: boolean) =>
    cn(
      'px-4 py-2 rounded-full text-sm font-medium transition-all border',
      selected
        ? 'bg-amber-400/20 border-amber-400/50 text-amber-300'
        : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:bg-zinc-700'
    );

  const choiceButtonClass = (selected: boolean) =>
    cn(
      'h-10 flex-1 rounded-xl text-sm font-medium transition-all border',
      selected
        ? 'bg-amber-400/20 border-amber-400/50 text-amber-300'
        : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:bg-zinc-700'
    );

  return (
    <div className="flex flex-col gap-5">
      <Card>
        <CardHeader>
          <CardTitle>친숙도</CardTitle>
        </CardHeader>
        <CardBody>
          <p className="text-sm text-zinc-400 mb-3">칵테일을 자주 드셔보셨나요?</p>
          <div className="flex gap-2">
            {([
              { value: 'beginner', label: '처음' },
              { value: 'casual', label: '가끔' },
              { value: 'experienced', label: '자주' },
            ] as const).map(({ value, label }) => (
              <button
                key={value}
                type="button"
                onClick={() => setPrefs((p) => ({ ...p, experience: value }))}
                className={choiceButtonClass(prefs.experience === value)}
              >
                {label}
              </button>
            ))}
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>선호 도수</CardTitle>
        </CardHeader>
        <CardBody>
          <div className="flex gap-2">
            {([
              { value: 'none', label: '무알콜' },
              { value: 'low', label: '약함' },
              { value: 'medium', label: '중간' },
              { value: 'high', label: '강함' },
            ] as const).map(({ value, label }) => (
              <button
                key={value}
                type="button"
                onClick={() => setPrefs((p) => ({ ...p, alcoholTolerance: value }))}
                className={choiceButtonClass(prefs.alcoholTolerance === value)}
              >
                {label}
              </button>
            ))}
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>
            선호 맛 태그{' '}
            <span className="text-sm text-zinc-500 font-normal">(여러 개 가능)</span>
          </CardTitle>
        </CardHeader>
        <CardBody>
          <div className="flex flex-wrap gap-2">
            {TASTE_TAGS.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() =>
                  setPrefs((p) => ({ ...p, tasteTags: toggleTag(p.tasteTags, t.id) }))
                }
                className={tagButtonClass(prefs.tasteTags.includes(t.id))}
              >
                {t.label}
              </button>
            ))}
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>
            선호 향 태그{' '}
            <span className="text-sm text-zinc-500 font-normal">(여러 개 가능)</span>
          </CardTitle>
        </CardHeader>
        <CardBody>
          <div className="flex flex-wrap gap-2">
            {AROMA_TAGS.map((a) => (
              <button
                key={a.id}
                type="button"
                onClick={() =>
                  setPrefs((p) => ({ ...p, aromaTags: toggleTag(p.aromaTags, a.id) }))
                }
                className={tagButtonClass(prefs.aromaTags.includes(a.id))}
              >
                {a.label}
              </button>
            ))}
          </div>
        </CardBody>
      </Card>

      <Button size="lg" className="w-full" disabled={!isComplete} onClick={handleSubmit}>
        다음 단계로 →
      </Button>
    </div>
  );
}

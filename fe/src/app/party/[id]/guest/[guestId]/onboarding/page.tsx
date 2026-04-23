'use client';

import { use } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { usePartyStore, useGuest } from '@/lib/store/partyStore';
import type { GuestPreferences } from '@/lib/types';
import { saveInitialTags } from '@/lib/api';
import { now } from '@/lib/utils';
import TasteForm from '@/components/onboarding/TasteForm';
import StepIndicator from '@/components/ui/StepIndicator';

const STEPS = [
  { label: '취향', icon: '🎯' },
  { label: '추가질문', icon: '💬' },
];

export default function OnboardingPage({
  params,
}: {
  params: Promise<{ id: string; guestId: string }>;
}) {
  const { id, guestId } = use(params);
  const router = useRouter();
  const guest = useGuest(id, guestId);
  const store = usePartyStore();

  if (!guest) {
    return (
      <main className="min-h-screen flex items-center justify-center px-4">
        <div className="text-center">
          <p className="text-zinc-400">게스트를 찾을 수 없어요</p>
          <Link href={`/party/${id}`} className="text-amber-400 text-sm mt-2 inline-block">
            파티로 돌아가기
          </Link>
        </div>
      </main>
    );
  }

  const handleTasteSubmit = async (prefs: GuestPreferences) => {
    store.setPreferences(id, guestId, prefs);
    store.addConversationEntry(id, guestId, {
      timestamp: now(),
      type: 'preference_input',
      content: `취향 입력 완료: 경험 ${prefs.experience}, 도수 ${prefs.alcoholTolerance}, 맛 [${prefs.tasteTags.join(', ')}], 향 [${prefs.aromaTags.join(', ')}]`,
    });
    try {
      if (guest.dbId) {
        await saveInitialTags(guest.dbId, prefs);
      }
    } catch {
      // DB 저장 실패해도 UI 흐름은 계속
    }
    router.push(`/party/${id}/guest/${guestId}/onboarding/chat`);
  };

  return (
    <main className="min-h-screen px-4 py-8 max-w-lg mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <Link href={`/party/${id}`} className="text-xs text-zinc-500 hover:text-zinc-300">
            ← 파티로
          </Link>
          <h1 className="text-xl font-bold text-zinc-100 mt-1">
            {guest.name}님의 취향 입력
          </h1>
        </div>
      </div>

      <div className="flex justify-center mb-8">
        <StepIndicator steps={STEPS} current={0} />
      </div>

      <TasteForm onSubmit={handleTasteSubmit} />
    </main>
  );
}

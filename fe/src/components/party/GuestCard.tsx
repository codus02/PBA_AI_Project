import Link from 'next/link';
import type { Guest } from '@/lib/types';
import Card from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';
import { cn } from '@/lib/utils';

const STEP_CONFIG: Record<
  Guest['step'],
  { label: string; badge: 'default' | 'blue' | 'amber' | 'green' | 'purple' | 'red'; path: string; emoji: string }
> = {
  onboarding: { label: '취향 입력 중', badge: 'default', path: 'onboarding', emoji: '📝' },
  tasting: { label: '시음 추천 완료', badge: 'blue', path: 'tasting', emoji: '🍹' },
  feedback: { label: '피드백 입력 중', badge: 'purple', path: 'tasting', emoji: '💬' },
  final: { label: '최종 추천 완료', badge: 'amber', path: 'final', emoji: '⭐' },
  brew: { label: '제조 중', badge: 'green', path: 'brew', emoji: '🌀' },
  complete: { label: '완료', badge: 'green', path: 'brew', emoji: '✅' },
};

interface GuestCardProps {
  guest: Guest;
  partyId: string;
}

export default function GuestCard({ guest, partyId }: GuestCardProps) {
  const config = STEP_CONFIG[guest.step];

  return (
    <Link href={`/party/${partyId}/guest/${guest.id}/${config.path}`}>
      <Card className={cn(
        'hover:border-zinc-700 hover:bg-zinc-800/80 transition-all cursor-pointer active:scale-[0.99]',
        guest.step === 'complete' && 'opacity-70'
      )}>
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center text-xl">
            {guest.tastingRecommendation?.imageEmoji ?? '👤'}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-0.5">
              <p className="font-semibold text-zinc-100">{guest.name}</p>
              {guest.satisfaction && (
                <span className="text-xs text-amber-400">{'⭐'.repeat(guest.satisfaction)}</span>
              )}
            </div>
            {guest.finalRecommendation ? (
              <p className="text-sm text-zinc-400 truncate">
                {guest.finalRecommendation.name}
              </p>
            ) : guest.tastingRecommendation ? (
              <p className="text-sm text-zinc-400 truncate">
                시음: {guest.tastingRecommendation.name}
              </p>
            ) : (
              <p className="text-sm text-zinc-500">취향 입력 대기 중</p>
            )}
          </div>
          <div className="flex flex-col items-end gap-1">
            <Badge variant={config.badge}>
              {config.emoji} {config.label}
            </Badge>
            <span className="text-xs text-zinc-600">→</span>
          </div>
        </div>
      </Card>
    </Link>
  );
}

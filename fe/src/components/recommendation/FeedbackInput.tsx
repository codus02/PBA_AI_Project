'use client';

import { useState } from 'react';
import Button from '@/components/ui/Button';
import Card, { CardHeader, CardTitle, CardBody } from '@/components/ui/Card';
import { Textarea } from '@/components/ui/Input';

interface FeedbackInputProps {
  onSubmit: (feedback: string) => void;
  loading?: boolean;
}

const QUICK_FEEDBACKS = [
  '너무 달아요',
  '좀 더 상큼했으면 좋겠어요',
  '쓴맛이 강해요',
  '알코올이 너무 강해요',
  '더 가볍게 해주세요',
  '향이 더 강했으면 좋겠어요',
  '딱 좋아요!',
];

export default function FeedbackInput({ onSubmit, loading = false }: FeedbackInputProps) {
  const [text, setText] = useState('');

  return (
    <Card>
      <CardHeader>
        <CardTitle>시음 피드백</CardTitle>
        <p className="text-sm text-zinc-400 mt-1">
          어떠셨나요? 자유롭게 말씀해주세요
        </p>
      </CardHeader>
      <CardBody className="flex flex-col gap-4">
        <div className="flex flex-wrap gap-2">
          {QUICK_FEEDBACKS.map((fb) => (
            <button
              key={fb}
              type="button"
              onClick={() => setText((prev) => prev ? `${prev} ${fb}` : fb)}
              className="px-3 py-1.5 rounded-full text-xs bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 transition-all"
            >
              {fb}
            </button>
          ))}
        </div>

        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="예: 너무 달고 알코올이 좀 강한 것 같아요. 좀 더 상큼한 맛이 있었으면 좋겠어요."
          rows={3}
        />

        <Button
          size="lg"
          className="w-full"
          disabled={!text.trim()}
          loading={loading}
          onClick={() => onSubmit(text.trim())}
        >
          피드백 반영해서 재추천 받기 →
        </Button>
      </CardBody>
    </Card>
  );
}

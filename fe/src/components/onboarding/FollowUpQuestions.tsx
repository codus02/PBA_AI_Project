'use client';

import { useState } from 'react';
import type { FollowUpQuestion, FollowUpAnswer } from '@/lib/types';
import Button from '@/components/ui/Button';
import Card, { CardHeader, CardTitle, CardBody } from '@/components/ui/Card';
import { cn } from '@/lib/utils';

interface FollowUpQuestionsProps {
  questions: FollowUpQuestion[];
  onSubmit: (answers: FollowUpAnswer[]) => void;
}

export default function FollowUpQuestions({ questions, onSubmit }: FollowUpQuestionsProps) {
  const [answers, setAnswers] = useState<Record<string, string>>({});

  const selectAnswer = (questionId: string, answer: string) => {
    setAnswers((prev) => ({ ...prev, [questionId]: answer }));
  };

  const allAnswered = questions.every((q) => answers[q.id]);

  const handleSubmit = () => {
    const result: FollowUpAnswer[] = questions.map((q) => ({
      questionId: q.id,
      question: q.question,
      answer: answers[q.id],
    }));
    onSubmit(result);
  };

  return (
    <div className="flex flex-col gap-5">
      <p className="text-sm text-zinc-400">
        좀 더 정확한 추천을 위해 몇 가지 추가 질문에 답해주세요
      </p>

      {questions.map((q, i) => (
        <Card key={q.id}>
          <CardHeader>
            <CardTitle className="text-base">
              <span className="text-amber-400 mr-2">Q{i + 1}.</span>
              {q.question}
            </CardTitle>
          </CardHeader>
          <CardBody>
            <div className="flex flex-col gap-2">
              {q.options.map((opt) => (
                <button
                  key={opt}
                  type="button"
                  onClick={() => selectAnswer(q.id, opt)}
                  className={cn(
                    'text-left px-4 py-3 rounded-xl text-sm transition-all border',
                    answers[q.id] === opt
                      ? 'bg-amber-400/20 border-amber-400/50 text-amber-300'
                      : 'bg-zinc-800 border-zinc-700 text-zinc-300 hover:bg-zinc-700'
                  )}
                >
                  {answers[q.id] === opt && <span className="mr-2">✓</span>}
                  {opt}
                </button>
              ))}
            </div>
          </CardBody>
        </Card>
      ))}

      <Button
        size="lg"
        className="w-full"
        disabled={!allAnswered}
        onClick={handleSubmit}
      >
        추천 받기 →
      </Button>
    </div>
  );
}

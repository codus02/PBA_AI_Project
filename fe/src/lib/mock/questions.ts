import type { FollowUpQuestion } from '@/lib/types';

export const MOCK_FOLLOWUP_QUESTIONS: FollowUpQuestion[] = [
  {
    id: 'occasion',
    question: '오늘 파티의 분위기는 어떤가요?',
    options: ['신나고 활기차게', '조용하고 세련되게', '편안하고 캐주얼하게', '특별하고 기억에 남게'],
  },
  {
    id: 'temperature',
    question: '음료의 온도는 어떻게 선호하세요?',
    options: ['얼음 듬뿍 차갑게', '약간 차갑게', '상온에 가깝게'],
  },
  {
    id: 'texture',
    question: '선호하는 음료 질감은?',
    options: ['탄산이 있는 청량감', '부드럽고 크리미하게', '걸쭉하게', '깔끔하고 가볍게'],
  },
  {
    id: 'base_spirit',
    question: '혹시 특별히 좋아하거나 싫어하는 주류가 있나요?',
    options: ['보드카 좋아요', '위스키 좋아요', '럼 좋아요', '진 좋아요', '테킬라 좋아요', '상관없어요'],
  },
  {
    id: 'garnish',
    question: '칵테일에 들어가는 과일/허브 향은 어떤가요?',
    options: ['시트러스 (레몬, 라임, 오렌지)', '베리류 (딸기, 블루베리)', '민트 / 바질', '없어도 좋아요'],
  },
];

export function getFollowUpQuestions(count = 3): FollowUpQuestion[] {
  // 무작위로 섞어서 count개 반환
  const shuffled = [...MOCK_FOLLOWUP_QUESTIONS].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, count);
}

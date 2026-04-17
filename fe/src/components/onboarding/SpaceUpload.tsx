'use client';

import { useState, useCallback } from 'react';
import type { SpaceAnalysis } from '@/lib/types';
import { analyzeSpaceImage } from '@/lib/mock/spaceAnalysis';
import Button from '@/components/ui/Button';
import Card, { CardHeader, CardTitle, CardBody } from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';

interface SpaceUploadProps {
  onComplete: (analysis: SpaceAnalysis, imageUrl: string) => void;
  onSkip: () => void;
}

export default function SpaceUpload({ onComplete, onSkip }: SpaceUploadProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<SpaceAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(async (file: File) => {
    const reader = new FileReader();
    reader.onload = async (e) => {
      const dataUrl = e.target?.result as string;
      setPreview(dataUrl);
      setLoading(true);
      try {
        const result = await analyzeSpaceImage(dataUrl);
        setAnalysis(result);
      } finally {
        setLoading(false);
      }
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith('image/')) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="flex flex-col gap-5">
      <Card>
        <CardHeader>
          <CardTitle>파티 공간 이미지 업로드</CardTitle>
          <p className="text-sm text-zinc-400 mt-1">
            파티 분위기에 맞는 칵테일을 추천해드려요
          </p>
        </CardHeader>
        <CardBody>
          {!preview ? (
            <label
              className={`flex flex-col items-center justify-center h-40 rounded-xl border-2 border-dashed cursor-pointer transition-all ${
                dragOver
                  ? 'border-amber-400 bg-amber-400/10'
                  : 'border-zinc-700 hover:border-zinc-500 hover:bg-zinc-800/50'
              }`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              <span className="text-4xl mb-2">📷</span>
              <span className="text-sm text-zinc-400">이미지를 드래그하거나 클릭해서 업로드</span>
              <span className="text-xs text-zinc-600 mt-1">JPG, PNG, WEBP 지원</span>
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFile(file);
                }}
              />
            </label>
          ) : (
            <div className="flex flex-col gap-4">
              <div className="relative rounded-xl overflow-hidden h-40">
                <img
                  src={preview}
                  alt="공간 이미지"
                  className="w-full h-full object-cover"
                />
                {loading && (
                  <div className="absolute inset-0 bg-zinc-950/70 flex flex-col items-center justify-center gap-2">
                    <div className="w-8 h-8 border-2 border-amber-400 border-t-transparent rounded-full animate-spin" />
                    <span className="text-sm text-amber-400">공간 분석 중...</span>
                  </div>
                )}
              </div>

              {analysis && (
                <div className="flex flex-col gap-3">
                  <div className="flex items-center gap-2">
                    <span className="text-green-400 text-sm">✓ 분석 완료</span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-zinc-800 rounded-xl p-3">
                      <p className="text-xs text-zinc-500 mb-1">스타일</p>
                      <p className="text-sm text-zinc-100 font-medium">{analysis.style}</p>
                    </div>
                    <div className="bg-zinc-800 rounded-xl p-3">
                      <p className="text-xs text-zinc-500 mb-1">분위기</p>
                      <p className="text-sm text-zinc-100 font-medium">{analysis.mood}</p>
                    </div>
                  </div>
                  <div className="bg-zinc-800 rounded-xl p-3">
                    <p className="text-xs text-zinc-500 mb-2">감지된 컬러</p>
                    <div className="flex flex-wrap gap-1.5">
                      {analysis.colors.map((c) => (
                        <Badge key={c} variant="amber">{c}</Badge>
                      ))}
                    </div>
                  </div>
                  <p className="text-sm text-zinc-400">{analysis.atmosphere}</p>
                </div>
              )}
            </div>
          )}
        </CardBody>
      </Card>
{/* 
      <div className="flex gap-3">
        <Button variant="secondary" className="flex-1" onClick={onSkip}>
          건너뛰기
        </Button>
        {analysis && preview && (
          <Button className="flex-1" onClick={() => onComplete(analysis, preview)}>
            다음 단계로 →
          </Button>
        )}
      </div> */}
    </div>
  );
}

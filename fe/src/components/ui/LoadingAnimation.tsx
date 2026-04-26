'use client';

import { useState, useEffect, useRef } from 'react';
import Image from 'next/image';

const FRAMES = [5, 6, 7, 8, 5, 6] as const;

interface LoadingAnimationProps {
  loop?: boolean;
  onComplete?: () => void;
  size?: number;
  className?: string;
}

export default function LoadingAnimation({
  loop = true,
  onComplete,
  size = 120,
  className = '',
}: LoadingAnimationProps) {
  const [frame, setFrame] = useState(0);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  useEffect(() => {
    let count = 0;
    const interval = setInterval(() => {
      count++;
      if (!loop && count >= FRAMES.length) {
        clearInterval(interval);
        onCompleteRef.current?.();
        return;
      }
      setFrame(count % FRAMES.length);
    }, 200);
    return () => clearInterval(interval);
  }, [loop]);

  return (
    <div className={`relative ${className}`} style={{ width: size, height: size }}>
      {FRAMES.map((n, i) => (
        <Image
          key={`${n}-${i}`}
          src={`/image_${n}.png`}
          alt=""
          width={size}
          height={size}
          priority
          className={`absolute inset-0 ${i === frame ? 'opacity-100' : 'opacity-0'}`}
          style={{ transition: 'none' }}
        />
      ))}
    </div>
  );
}

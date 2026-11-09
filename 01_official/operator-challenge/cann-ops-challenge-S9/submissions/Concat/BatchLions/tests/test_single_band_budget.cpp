// 独立累加每路实际补齐空间，检查单带紧预算及 int8 转换副本。
// Copyright (c) 2026 ascend_simula contributors
// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <algorithm>
#include <random>
#include "../op_host/s9_concat_layout.h"
int main(){
 std::mt19937 rng(111); unsigned checked=0;
 for(unsigned test=0;test<18000;++test){
  const unsigned esz=1u<<(test%3), k=1+rng()%256, rows=1+rng()%4095;
  unsigned ns[256],bw=0,mx=0,nz=0;
  for(unsigned i=0;i<k;++i){ns[i]=rng()%129; bw+=ns[i];mx=std::max(mx,ns[i]);nz+=ns[i]!=0;}
  if(!bw)continue;
  const unsigned gb=1+rng()%s9::S9L_BUF_N, ab=1+rng()%s9::S9L_BUF_N, dm=rng()%3;
  const auto l=s9::PlanConcatStage(bw,esz,k,mx,nz,gb,ab,0,rows,0,dm,0,true);
  if(!l.fits)continue;
  unsigned long long actual=0;
  for(unsigned i=0;i<k;++i)actual+=((unsigned long long)l.tileG*ns[i]*esz+31)/32*32;
  if(actual>l.stageSpan || (esz==1&&actual*2>l.castSpan) || l.totalBytes>l.limitBytes){
   std::printf("FAIL test=%u esz=%u actual=%llu stage=%u cast=%u\n",test,esz,actual,l.stageSpan,l.castSpan);return 1;
  }
  for(unsigned b=0;b<gb;++b)if(!l.StgBlockOk(b)||l.StgBase(b)%32)return 2;
  ++checked;
 }
 std::printf("PASS %u feasible randomized staging layouts, including int8 cast capacity and buffer alignment\n",checked);
}

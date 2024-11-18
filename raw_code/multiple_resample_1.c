static int multiple_resample(ResampleContext *c, AudioData *dst, int dst_size, AudioData *src, int src_size, int *consumed){
    int i, ret= -1;
    int av_unused mm_flags = av_get_cpu_flags();
    int need_emms= 0;
    if (c->compensation_distance)
        dst_size = FFMIN(dst_size, c->compensation_distance);
    for(i=0; i<dst->ch_count; i++){
#if HAVE_MMXEXT_INLINE
#if HAVE_SSE2_INLINE
             if(c->format == AV_SAMPLE_FMT_S16P && (mm_flags&AV_CPU_FLAG_SSE2)) ret= swri_resample_int16_sse2 (c, (int16_t*)dst->ch[i], (const int16_t*)src->ch[i], consumed, src_size, dst_size, i+1==dst->ch_count);
        else
#endif
             if(c->format == AV_SAMPLE_FMT_S16P && (mm_flags&AV_CPU_FLAG_MMX2 )){
                 ret= swri_resample_int16_mmx2 (c, (int16_t*)dst->ch[i], (const int16_t*)src->ch[i], consumed, src_size, dst_size, i+1==dst->ch_count);
                 need_emms= 1;
             } else
#endif
             if(c->format == AV_SAMPLE_FMT_S16P) ret= swri_resample_int16(c, (int16_t*)dst->ch[i], (const int16_t*)src->ch[i], consumed, src_size, dst_size, i+1==dst->ch_count);
        else if(c->format == AV_SAMPLE_FMT_S32P) ret= swri_resample_int32(c, (int32_t*)dst->ch[i], (const int32_t*)src->ch[i], consumed, src_size, dst_size, i+1==dst->ch_count);
#if HAVE_AVX_INLINE
        else if(c->format == AV_SAMPLE_FMT_FLTP && (mm_flags&AV_CPU_FLAG_AVX))
                                                 ret= swri_resample_float_avx (c, (float*)dst->ch[i], (const float*)src->ch[i], consumed, src_size, dst_size, i+1==dst->ch_count);
#endif
#if HAVE_SSE_INLINE
        else if(c->format == AV_SAMPLE_FMT_FLTP && (mm_flags&AV_CPU_FLAG_SSE))
                                                 ret= swri_resample_float_sse (c, (float*)dst->ch[i], (const float*)src->ch[i], consumed, src_size, dst_size, i+1==dst->ch_count);
#endif
        else if(c->format == AV_SAMPLE_FMT_FLTP) ret= swri_resample_float(c, (float  *)dst->ch[i], (const float  *)src->ch[i], consumed, src_size, dst_size, i+1==dst->ch_count);
#if HAVE_SSE2_INLINE
        else if(c->format == AV_SAMPLE_FMT_DBLP && (mm_flags&AV_CPU_FLAG_SSE2))
                                                 ret= swri_resample_double_sse2(c,(double *)dst->ch[i], (const double *)src->ch[i], consumed, src_size, dst_size, i+1==dst->ch_count);
#endif
        else if(c->format == AV_SAMPLE_FMT_DBLP) ret= swri_resample_double(c,(double *)dst->ch[i], (const double *)src->ch[i], consumed, src_size, dst_size, i+1==dst->ch_count);
    if(need_emms)
        emms_c();
    return ret;
void ff_fmt_convert_init_x86(FmtConvertContext *c, AVCodecContext *avctx)

{

    int mm_flags = av_get_cpu_flags();



    if (mm_flags & AV_CPU_FLAG_MMX) {

#if HAVE_YASM

        c->float_interleave = float_interleave_mmx;



        if(mm_flags & AV_CPU_FLAG_3DNOW){

            if(!(avctx->flags & CODEC_FLAG_BITEXACT)){

                c->float_to_int16 = ff_float_to_int16_3dnow;

                c->float_to_int16_interleave = float_to_int16_interleave_3dnow;

            }

        }

        if(mm_flags & AV_CPU_FLAG_3DNOWEXT){

            if(!(avctx->flags & CODEC_FLAG_BITEXACT)){

                c->float_to_int16_interleave = float_to_int16_interleave_3dn2;

            }

        }

#endif

        if(mm_flags & AV_CPU_FLAG_SSE){

            c->int32_to_float_fmul_scalar = int32_to_float_fmul_scalar_sse;

#if HAVE_YASM

            c->float_to_int16 = ff_float_to_int16_sse;

            c->float_to_int16_interleave = float_to_int16_interleave_sse;

            c->float_interleave = float_interleave_sse;

#endif

        }

        if(mm_flags & AV_CPU_FLAG_SSE2){

            c->int32_to_float_fmul_scalar = int32_to_float_fmul_scalar_sse2;

#if HAVE_YASM

            c->float_to_int16 = ff_float_to_int16_sse2;

            c->float_to_int16_interleave = float_to_int16_interleave_sse2;

#endif

        }

    }

}

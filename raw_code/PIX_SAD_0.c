PIX_SAD(mmxext)



#endif /* HAVE_INLINE_ASM */



av_cold void ff_dsputil_init_pix_mmx(DSPContext *c, AVCodecContext *avctx)

{

#if HAVE_INLINE_ASM

    int cpu_flags = av_get_cpu_flags();



    if (INLINE_MMX(cpu_flags)) {

        c->pix_abs[0][0] = sad16_mmx;

        c->pix_abs[0][1] = sad16_x2_mmx;

        c->pix_abs[0][2] = sad16_y2_mmx;

        c->pix_abs[0][3] = sad16_xy2_mmx;

        c->pix_abs[1][0] = sad8_mmx;

        c->pix_abs[1][1] = sad8_x2_mmx;

        c->pix_abs[1][2] = sad8_y2_mmx;

        c->pix_abs[1][3] = sad8_xy2_mmx;



        c->sad[0] = sad16_mmx;

        c->sad[1] = sad8_mmx;

    }

    if (INLINE_MMXEXT(cpu_flags)) {

        c->pix_abs[0][0] = sad16_mmxext;

        c->pix_abs[1][0] = sad8_mmxext;



        c->sad[0] = sad16_mmxext;

        c->sad[1] = sad8_mmxext;



        if (!(avctx->flags & CODEC_FLAG_BITEXACT)) {

            c->pix_abs[0][1] = sad16_x2_mmxext;

            c->pix_abs[0][2] = sad16_y2_mmxext;

            c->pix_abs[0][3] = sad16_xy2_mmxext;

            c->pix_abs[1][1] = sad8_x2_mmxext;

            c->pix_abs[1][2] = sad8_y2_mmxext;

            c->pix_abs[1][3] = sad8_xy2_mmxext;

        }

    }

    if (INLINE_SSE2(cpu_flags) && !(cpu_flags & AV_CPU_FLAG_3DNOW) && avctx->codec_id != AV_CODEC_ID_SNOW) {

        c->sad[0] = sad16_sse2;

    }

#endif /* HAVE_INLINE_ASM */

}

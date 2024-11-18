QPEL_H264(put_,        PUT_OP, mmxext)

QPEL_H264(avg_, AVG_MMXEXT_OP, mmxext)

QPEL_H264_V_XMM(put_,       PUT_OP, sse2)

QPEL_H264_V_XMM(avg_,AVG_MMXEXT_OP, sse2)

QPEL_H264_HV_XMM(put_,       PUT_OP, sse2)

QPEL_H264_HV_XMM(avg_,AVG_MMXEXT_OP, sse2)

QPEL_H264_H_XMM(put_,       PUT_OP, ssse3)

QPEL_H264_H_XMM(avg_,AVG_MMXEXT_OP, ssse3)

QPEL_H264_HV_XMM(put_,       PUT_OP, ssse3)

QPEL_H264_HV_XMM(avg_,AVG_MMXEXT_OP, ssse3)

#undef PAVGB



H264_MC_4816(mmxext)

H264_MC_816(H264_MC_V, sse2)

H264_MC_816(H264_MC_HV, sse2)

H264_MC_816(H264_MC_H, ssse3)

H264_MC_816(H264_MC_HV, ssse3)





//10bit

#define LUMA_MC_OP(OP, NUM, DEPTH, TYPE, OPT) \

void ff_ ## OP ## _h264_qpel ## NUM ## _ ## TYPE ## _ ## DEPTH ## _ ## OPT \

    (uint8_t *dst, uint8_t *src, int stride);



#define LUMA_MC_ALL(DEPTH, TYPE, OPT) \

    LUMA_MC_OP(put,  4, DEPTH, TYPE, OPT) \

    LUMA_MC_OP(avg,  4, DEPTH, TYPE, OPT) \

    LUMA_MC_OP(put,  8, DEPTH, TYPE, OPT) \

    LUMA_MC_OP(avg,  8, DEPTH, TYPE, OPT) \

    LUMA_MC_OP(put, 16, DEPTH, TYPE, OPT) \

    LUMA_MC_OP(avg, 16, DEPTH, TYPE, OPT)



#define LUMA_MC_816(DEPTH, TYPE, OPT) \

    LUMA_MC_OP(put,  8, DEPTH, TYPE, OPT) \

    LUMA_MC_OP(avg,  8, DEPTH, TYPE, OPT) \

    LUMA_MC_OP(put, 16, DEPTH, TYPE, OPT) \

    LUMA_MC_OP(avg, 16, DEPTH, TYPE, OPT)



LUMA_MC_ALL(10, mc00, mmxext)

LUMA_MC_ALL(10, mc10, mmxext)

LUMA_MC_ALL(10, mc20, mmxext)

LUMA_MC_ALL(10, mc30, mmxext)

LUMA_MC_ALL(10, mc01, mmxext)

LUMA_MC_ALL(10, mc11, mmxext)

LUMA_MC_ALL(10, mc21, mmxext)

LUMA_MC_ALL(10, mc31, mmxext)

LUMA_MC_ALL(10, mc02, mmxext)

LUMA_MC_ALL(10, mc12, mmxext)

LUMA_MC_ALL(10, mc22, mmxext)

LUMA_MC_ALL(10, mc32, mmxext)

LUMA_MC_ALL(10, mc03, mmxext)

LUMA_MC_ALL(10, mc13, mmxext)

LUMA_MC_ALL(10, mc23, mmxext)

LUMA_MC_ALL(10, mc33, mmxext)



LUMA_MC_816(10, mc00, sse2)

LUMA_MC_816(10, mc10, sse2)

LUMA_MC_816(10, mc10, sse2_cache64)

LUMA_MC_816(10, mc10, ssse3_cache64)

LUMA_MC_816(10, mc20, sse2)

LUMA_MC_816(10, mc20, sse2_cache64)

LUMA_MC_816(10, mc20, ssse3_cache64)

LUMA_MC_816(10, mc30, sse2)

LUMA_MC_816(10, mc30, sse2_cache64)

LUMA_MC_816(10, mc30, ssse3_cache64)

LUMA_MC_816(10, mc01, sse2)

LUMA_MC_816(10, mc11, sse2)

LUMA_MC_816(10, mc21, sse2)

LUMA_MC_816(10, mc31, sse2)

LUMA_MC_816(10, mc02, sse2)

LUMA_MC_816(10, mc12, sse2)

LUMA_MC_816(10, mc22, sse2)

LUMA_MC_816(10, mc32, sse2)

LUMA_MC_816(10, mc03, sse2)

LUMA_MC_816(10, mc13, sse2)

LUMA_MC_816(10, mc23, sse2)

LUMA_MC_816(10, mc33, sse2)



#define QPEL16_OPMC(OP, MC, MMX)\

void ff_ ## OP ## _h264_qpel16_ ## MC ## _10_ ## MMX(uint8_t *dst, uint8_t *src, int stride){\

    ff_ ## OP ## _h264_qpel8_ ## MC ## _10_ ## MMX(dst   , src   , stride);\

    ff_ ## OP ## _h264_qpel8_ ## MC ## _10_ ## MMX(dst+16, src+16, stride);\

    src += 8*stride;\

    dst += 8*stride;\

    ff_ ## OP ## _h264_qpel8_ ## MC ## _10_ ## MMX(dst   , src   , stride);\

    ff_ ## OP ## _h264_qpel8_ ## MC ## _10_ ## MMX(dst+16, src+16, stride);\

}



#define QPEL16_OP(MC, MMX)\

QPEL16_OPMC(put, MC, MMX)\

QPEL16_OPMC(avg, MC, MMX)



#define QPEL16(MMX)\

QPEL16_OP(mc00, MMX)\

QPEL16_OP(mc01, MMX)\

QPEL16_OP(mc02, MMX)\

QPEL16_OP(mc03, MMX)\

QPEL16_OP(mc10, MMX)\

QPEL16_OP(mc11, MMX)\

QPEL16_OP(mc12, MMX)\

QPEL16_OP(mc13, MMX)\

QPEL16_OP(mc20, MMX)\

QPEL16_OP(mc21, MMX)\

QPEL16_OP(mc22, MMX)\

QPEL16_OP(mc23, MMX)\

QPEL16_OP(mc30, MMX)\

QPEL16_OP(mc31, MMX)\

QPEL16_OP(mc32, MMX)\

QPEL16_OP(mc33, MMX)



#if ARCH_X86_32 && HAVE_YASM && CONFIG_H264QPEL // ARCH_X86_64 implies SSE2+

QPEL16(mmxext)

#endif



#endif /* HAVE_YASM */



#define SET_QPEL_FUNCS(PFX, IDX, SIZE, CPU, PREFIX)                          \

    do {                                                                     \

    c->PFX ## _pixels_tab[IDX][ 0] = PREFIX ## PFX ## SIZE ## _mc00_ ## CPU; \

    c->PFX ## _pixels_tab[IDX][ 1] = PREFIX ## PFX ## SIZE ## _mc10_ ## CPU; \

    c->PFX ## _pixels_tab[IDX][ 2] = PREFIX ## PFX ## SIZE ## _mc20_ ## CPU; \

    c->PFX ## _pixels_tab[IDX][ 3] = PREFIX ## PFX ## SIZE ## _mc30_ ## CPU; \

    c->PFX ## _pixels_tab[IDX][ 4] = PREFIX ## PFX ## SIZE ## _mc01_ ## CPU; \

    c->PFX ## _pixels_tab[IDX][ 5] = PREFIX ## PFX ## SIZE ## _mc11_ ## CPU; \

    c->PFX ## _pixels_tab[IDX][ 6] = PREFIX ## PFX ## SIZE ## _mc21_ ## CPU; \

    c->PFX ## _pixels_tab[IDX][ 7] = PREFIX ## PFX ## SIZE ## _mc31_ ## CPU; \

    c->PFX ## _pixels_tab[IDX][ 8] = PREFIX ## PFX ## SIZE ## _mc02_ ## CPU; \

    c->PFX ## _pixels_tab[IDX][ 9] = PREFIX ## PFX ## SIZE ## _mc12_ ## CPU; \

    c->PFX ## _pixels_tab[IDX][10] = PREFIX ## PFX ## SIZE ## _mc22_ ## CPU; \

    c->PFX ## _pixels_tab[IDX][11] = PREFIX ## PFX ## SIZE ## _mc32_ ## CPU; \

    c->PFX ## _pixels_tab[IDX][12] = PREFIX ## PFX ## SIZE ## _mc03_ ## CPU; \

    c->PFX ## _pixels_tab[IDX][13] = PREFIX ## PFX ## SIZE ## _mc13_ ## CPU; \

    c->PFX ## _pixels_tab[IDX][14] = PREFIX ## PFX ## SIZE ## _mc23_ ## CPU; \

    c->PFX ## _pixels_tab[IDX][15] = PREFIX ## PFX ## SIZE ## _mc33_ ## CPU; \

    } while (0)



#define H264_QPEL_FUNCS(x, y, CPU)                                                            \

    do {                                                                                      \

        c->put_h264_qpel_pixels_tab[0][x + y * 4] = put_h264_qpel16_mc ## x ## y ## _ ## CPU; \

        c->put_h264_qpel_pixels_tab[1][x + y * 4] = put_h264_qpel8_mc  ## x ## y ## _ ## CPU; \

        c->avg_h264_qpel_pixels_tab[0][x + y * 4] = avg_h264_qpel16_mc ## x ## y ## _ ## CPU; \

        c->avg_h264_qpel_pixels_tab[1][x + y * 4] = avg_h264_qpel8_mc  ## x ## y ## _ ## CPU; \

    } while (0)



#define H264_QPEL_FUNCS_10(x, y, CPU)                                                               \

    do {                                                                                            \

        c->put_h264_qpel_pixels_tab[0][x + y * 4] = ff_put_h264_qpel16_mc ## x ## y ## _10_ ## CPU; \

        c->put_h264_qpel_pixels_tab[1][x + y * 4] = ff_put_h264_qpel8_mc  ## x ## y ## _10_ ## CPU; \

        c->avg_h264_qpel_pixels_tab[0][x + y * 4] = ff_avg_h264_qpel16_mc ## x ## y ## _10_ ## CPU; \

        c->avg_h264_qpel_pixels_tab[1][x + y * 4] = ff_avg_h264_qpel8_mc  ## x ## y ## _10_ ## CPU; \

    } while (0)



void ff_h264qpel_init_x86(H264QpelContext *c, int bit_depth)

{

    int high_bit_depth = bit_depth > 8;

    int mm_flags = av_get_cpu_flags();



#if HAVE_MMXEXT_EXTERNAL

    if (!high_bit_depth) {

        SET_QPEL_FUNCS(put_h264_qpel, 0, 16, mmxext, );

        SET_QPEL_FUNCS(put_h264_qpel, 1,  8, mmxext, );

        SET_QPEL_FUNCS(put_h264_qpel, 2,  4, mmxext, );

        SET_QPEL_FUNCS(avg_h264_qpel, 0, 16, mmxext, );

        SET_QPEL_FUNCS(avg_h264_qpel, 1,  8, mmxext, );

        SET_QPEL_FUNCS(avg_h264_qpel, 2,  4, mmxext, );

    } else if (bit_depth == 10) {

#if !ARCH_X86_64

        SET_QPEL_FUNCS(avg_h264_qpel, 0, 16, 10_mmxext, ff_);

        SET_QPEL_FUNCS(put_h264_qpel, 0, 16, 10_mmxext, ff_);

        SET_QPEL_FUNCS(put_h264_qpel, 1,  8, 10_mmxext, ff_);

        SET_QPEL_FUNCS(avg_h264_qpel, 1,  8, 10_mmxext, ff_);

#endif

        SET_QPEL_FUNCS(put_h264_qpel, 2, 4,  10_mmxext, ff_);

        SET_QPEL_FUNCS(avg_h264_qpel, 2, 4,  10_mmxext, ff_);

    }

#endif



#if HAVE_SSE2_EXTERNAL

    if (!(mm_flags & AV_CPU_FLAG_SSE2SLOW) && !high_bit_depth) {

        // these functions are slower than mmx on AMD, but faster on Intel

        H264_QPEL_FUNCS(0, 0, sse2);

    }



    if (!high_bit_depth) {

        H264_QPEL_FUNCS(0, 1, sse2);

        H264_QPEL_FUNCS(0, 2, sse2);

        H264_QPEL_FUNCS(0, 3, sse2);

        H264_QPEL_FUNCS(1, 1, sse2);

        H264_QPEL_FUNCS(1, 2, sse2);

        H264_QPEL_FUNCS(1, 3, sse2);

        H264_QPEL_FUNCS(2, 1, sse2);

        H264_QPEL_FUNCS(2, 2, sse2);

        H264_QPEL_FUNCS(2, 3, sse2);

        H264_QPEL_FUNCS(3, 1, sse2);

        H264_QPEL_FUNCS(3, 2, sse2);

        H264_QPEL_FUNCS(3, 3, sse2);

    }



    if (bit_depth == 10) {

        SET_QPEL_FUNCS(put_h264_qpel, 0, 16, 10_sse2, ff_);

        SET_QPEL_FUNCS(put_h264_qpel, 1,  8, 10_sse2, ff_);

        SET_QPEL_FUNCS(avg_h264_qpel, 0, 16, 10_sse2, ff_);

        SET_QPEL_FUNCS(avg_h264_qpel, 1,  8, 10_sse2, ff_);

        H264_QPEL_FUNCS_10(1, 0, sse2_cache64);

        H264_QPEL_FUNCS_10(2, 0, sse2_cache64);

        H264_QPEL_FUNCS_10(3, 0, sse2_cache64);

    }

#endif



#if HAVE_SSSE3_EXTERNAL

    if (!high_bit_depth) {

        H264_QPEL_FUNCS(1, 0, ssse3);

        H264_QPEL_FUNCS(1, 1, ssse3);

        H264_QPEL_FUNCS(1, 2, ssse3);

        H264_QPEL_FUNCS(1, 3, ssse3);

        H264_QPEL_FUNCS(2, 0, ssse3);

        H264_QPEL_FUNCS(2, 1, ssse3);

        H264_QPEL_FUNCS(2, 2, ssse3);

        H264_QPEL_FUNCS(2, 3, ssse3);

        H264_QPEL_FUNCS(3, 0, ssse3);

        H264_QPEL_FUNCS(3, 1, ssse3);

        H264_QPEL_FUNCS(3, 2, ssse3);

        H264_QPEL_FUNCS(3, 3, ssse3);

    }



    if (bit_depth == 10) {

        H264_QPEL_FUNCS_10(1, 0, ssse3_cache64);

        H264_QPEL_FUNCS_10(2, 0, ssse3_cache64);

        H264_QPEL_FUNCS_10(3, 0, ssse3_cache64);

    }

#endif



#if HAVE_AVX_EXTERNAL

    if (bit_depth == 10) {

        H264_QPEL_FUNCS_10(1, 0, sse2);

        H264_QPEL_FUNCS_10(2, 0, sse2);

        H264_QPEL_FUNCS_10(3, 0, sse2);

    }

#endif

}

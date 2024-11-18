void ff_avg_dirac_pixels32_sse2(uint8_t *dst, const uint8_t *src[5], int stride, int h)

{

    if (h&3) {

        ff_avg_dirac_pixels32_c(dst, src, stride, h);

    } else {

    ff_avg_pixels16_sse2(dst   , src[0]   , stride, h);

    ff_avg_pixels16_sse2(dst+16, src[0]+16, stride, h);

    }

}

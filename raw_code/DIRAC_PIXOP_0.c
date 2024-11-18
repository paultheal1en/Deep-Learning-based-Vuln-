DIRAC_PIXOP(put, ff_put, mmx)

DIRAC_PIXOP(avg, ff_avg, mmx)

DIRAC_PIXOP(avg, ff_avg, mmxext)



void ff_put_dirac_pixels16_sse2(uint8_t *dst, const uint8_t *src[5], int stride, int h)

{

    if (h&3)

        ff_put_dirac_pixels16_c(dst, src, stride, h);

    else

    ff_put_pixels16_sse2(dst, src[0], stride, h);

}

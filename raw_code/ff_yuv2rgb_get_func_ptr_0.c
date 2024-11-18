SwsFunc ff_yuv2rgb_get_func_ptr(SwsContext *c)

{

    SwsFunc t = NULL;

#if (HAVE_MMX2 || HAVE_MMX) && CONFIG_GPL

     t = ff_yuv2rgb_init_mmx(c);

#endif

#if HAVE_VIS

    t = ff_yuv2rgb_init_vis(c);

#endif

#if CONFIG_MLIB

    t = ff_yuv2rgb_init_mlib(c);

#endif

#if HAVE_ALTIVEC && CONFIG_GPL

    if (c->flags & SWS_CPU_CAPS_ALTIVEC)

        t = ff_yuv2rgb_init_altivec(c);

#endif



#if ARCH_BFIN

    if (c->flags & SWS_CPU_CAPS_BFIN)

        t = ff_yuv2rgb_get_func_ptr_bfin(c);

#endif



    if (t)

        return t;



    av_log(c, AV_LOG_WARNING, "No accelerated colorspace conversion found.\n");



    switch (c->dstFormat) {

    case PIX_FMT_RGB48BE:

    case PIX_FMT_RGB48LE:    return yuv2rgb_c_48;

    case PIX_FMT_ARGB:

    case PIX_FMT_ABGR:       if (CONFIG_SWSCALE_ALPHA && c->srcFormat == PIX_FMT_YUVA420P) return yuva2argb_c;

    case PIX_FMT_RGBA:

    case PIX_FMT_BGRA:       return (CONFIG_SWSCALE_ALPHA && c->srcFormat == PIX_FMT_YUVA420P) ? yuva2rgba_c : yuv2rgb_c_32;

    case PIX_FMT_RGB24:      return yuv2rgb_c_24_rgb;

    case PIX_FMT_BGR24:      return yuv2rgb_c_24_bgr;

    case PIX_FMT_RGB565:

    case PIX_FMT_BGR565:

    case PIX_FMT_RGB555:

    case PIX_FMT_BGR555:     return yuv2rgb_c_16;

    case PIX_FMT_RGB8:

    case PIX_FMT_BGR8:       return yuv2rgb_c_8_ordered_dither;

    case PIX_FMT_RGB4:

    case PIX_FMT_BGR4:       return yuv2rgb_c_4_ordered_dither;

    case PIX_FMT_RGB4_BYTE:

    case PIX_FMT_BGR4_BYTE:  return yuv2rgb_c_4b_ordered_dither;

    case PIX_FMT_MONOBLACK:  return yuv2rgb_c_1_ordered_dither;

    default:

        assert(0);

    }

    return NULL;

}

av_cold void ff_vp9_init_static(AVCodec *codec)

{

    if (    vpx_codec_version_major() < 1

        || (vpx_codec_version_major() == 1 && vpx_codec_version_minor() < 3))

        codec->capabilities |= AV_CODEC_CAP_EXPERIMENTAL;

    codec->pix_fmts = vp9_pix_fmts_def;

#if CONFIG_LIBVPX_VP9_ENCODER

    if (    vpx_codec_version_major() > 1

        || (vpx_codec_version_major() == 1 && vpx_codec_version_minor() >= 4)) {

#ifdef VPX_CODEC_CAP_HIGHBITDEPTH

        vpx_codec_caps_t codec_caps = vpx_codec_get_caps(vpx_codec_vp9_cx());

        if (codec_caps & VPX_CODEC_CAP_HIGHBITDEPTH)

            codec->pix_fmts = vp9_pix_fmts_highbd;

        else

#endif

            codec->pix_fmts = vp9_pix_fmts_highcol;

    }

#endif

}

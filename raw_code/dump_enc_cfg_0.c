static av_cold void dump_enc_cfg(AVCodecContext *avctx,

                                 const struct vpx_codec_enc_cfg *cfg)

{

    int width = -30;

    int level = AV_LOG_DEBUG;



    av_log(avctx, level, "vpx_codec_enc_cfg\n");

    av_log(avctx, level, "generic settings\n"

           "  %*s%u\n  %*s%u\n  %*s%u\n  %*s%u\n  %*s%u\n"

#if CONFIG_LIBVPX_VP9_ENCODER && defined(VPX_IMG_FMT_HIGHBITDEPTH)

           "  %*s%u\n  %*s%u\n"

#endif

           "  %*s{%u/%u}\n  %*s%u\n  %*s%d\n  %*s%u\n",

           width, "g_usage:",           cfg->g_usage,

           width, "g_threads:",         cfg->g_threads,

           width, "g_profile:",         cfg->g_profile,

           width, "g_w:",               cfg->g_w,

           width, "g_h:",               cfg->g_h,

#if CONFIG_LIBVPX_VP9_ENCODER && defined(VPX_IMG_FMT_HIGHBITDEPTH)

           width, "g_bit_depth:",       cfg->g_bit_depth,

           width, "g_input_bit_depth:", cfg->g_input_bit_depth,

#endif

           width, "g_timebase:",        cfg->g_timebase.num, cfg->g_timebase.den,

           width, "g_error_resilient:", cfg->g_error_resilient,

           width, "g_pass:",            cfg->g_pass,

           width, "g_lag_in_frames:",   cfg->g_lag_in_frames);

    av_log(avctx, level, "rate control settings\n"

           "  %*s%u\n  %*s%u\n  %*s%u\n  %*s%u\n"

           "  %*s%d\n  %*s%p(%"SIZE_SPECIFIER")\n  %*s%u\n",

           width, "rc_dropframe_thresh:",   cfg->rc_dropframe_thresh,

           width, "rc_resize_allowed:",     cfg->rc_resize_allowed,

           width, "rc_resize_up_thresh:",   cfg->rc_resize_up_thresh,

           width, "rc_resize_down_thresh:", cfg->rc_resize_down_thresh,

           width, "rc_end_usage:",          cfg->rc_end_usage,

           width, "rc_twopass_stats_in:",   cfg->rc_twopass_stats_in.buf, cfg->rc_twopass_stats_in.sz,

           width, "rc_target_bitrate:",     cfg->rc_target_bitrate);

    av_log(avctx, level, "quantizer settings\n"

           "  %*s%u\n  %*s%u\n",

           width, "rc_min_quantizer:", cfg->rc_min_quantizer,

           width, "rc_max_quantizer:", cfg->rc_max_quantizer);

    av_log(avctx, level, "bitrate tolerance\n"

           "  %*s%u\n  %*s%u\n",

           width, "rc_undershoot_pct:", cfg->rc_undershoot_pct,

           width, "rc_overshoot_pct:",  cfg->rc_overshoot_pct);

    av_log(avctx, level, "decoder buffer model\n"

            "  %*s%u\n  %*s%u\n  %*s%u\n",

            width, "rc_buf_sz:",         cfg->rc_buf_sz,

            width, "rc_buf_initial_sz:", cfg->rc_buf_initial_sz,

            width, "rc_buf_optimal_sz:", cfg->rc_buf_optimal_sz);

    av_log(avctx, level, "2 pass rate control settings\n"

           "  %*s%u\n  %*s%u\n  %*s%u\n",

           width, "rc_2pass_vbr_bias_pct:",       cfg->rc_2pass_vbr_bias_pct,

           width, "rc_2pass_vbr_minsection_pct:", cfg->rc_2pass_vbr_minsection_pct,

           width, "rc_2pass_vbr_maxsection_pct:", cfg->rc_2pass_vbr_maxsection_pct);

    av_log(avctx, level, "keyframing settings\n"

           "  %*s%d\n  %*s%u\n  %*s%u\n",

           width, "kf_mode:",     cfg->kf_mode,

           width, "kf_min_dist:", cfg->kf_min_dist,

           width, "kf_max_dist:", cfg->kf_max_dist);

    av_log(avctx, level, "\n");

}

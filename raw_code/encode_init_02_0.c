static av_cold int encode_init(AVCodecContext* avc_context)

{

    th_info t_info;

    th_comment t_comment;

    ogg_packet o_packet;

    unsigned int offset;

    TheoraContext *h = avc_context->priv_data;

    uint32_t gop_size = avc_context->gop_size;



    /* Set up the theora_info struct */

    th_info_init(&t_info);

    t_info.frame_width  = FFALIGN(avc_context->width,  16);

    t_info.frame_height = FFALIGN(avc_context->height, 16);

    t_info.pic_width    = avc_context->width;

    t_info.pic_height   = avc_context->height;

    t_info.pic_x        = 0;

    t_info.pic_y        = 0;

    /* Swap numerator and denominator as time_base in AVCodecContext gives the

     * time period between frames, but theora_info needs the framerate.  */

    t_info.fps_numerator   = avc_context->time_base.den;

    t_info.fps_denominator = avc_context->time_base.num;

    if (avc_context->sample_aspect_ratio.num) {

        t_info.aspect_numerator   = avc_context->sample_aspect_ratio.num;

        t_info.aspect_denominator = avc_context->sample_aspect_ratio.den;

    } else {

        t_info.aspect_numerator   = 1;

        t_info.aspect_denominator = 1;

    }



    if (avc_context->color_primaries == AVCOL_PRI_BT470M)

        t_info.colorspace = TH_CS_ITU_REC_470M;

    else if (avc_context->color_primaries == AVCOL_PRI_BT470BG)

        t_info.colorspace = TH_CS_ITU_REC_470BG;

    else

        t_info.colorspace = TH_CS_UNSPECIFIED;



    if (avc_context->pix_fmt == AV_PIX_FMT_YUV420P)

        t_info.pixel_fmt = TH_PF_420;

    else if (avc_context->pix_fmt == AV_PIX_FMT_YUV422P)

        t_info.pixel_fmt = TH_PF_422;

    else if (avc_context->pix_fmt == AV_PIX_FMT_YUV444P)

        t_info.pixel_fmt = TH_PF_444;

    else {

        av_log(avc_context, AV_LOG_ERROR, "Unsupported pix_fmt\n");

        return -1;

    }

    av_pix_fmt_get_chroma_sub_sample(avc_context->pix_fmt,

                                     &h->uv_hshift, &h->uv_vshift);



    if (avc_context->flags & CODEC_FLAG_QSCALE) {

        /* to be constant with the libvorbis implementation, clip global_quality to 0 - 10

           Theora accepts a quality parameter p, which is:

                * 0 <= p <=63

                * an int value

         */

        t_info.quality        = av_clipf(avc_context->global_quality / (float)FF_QP2LAMBDA, 0, 10) * 6.3;

        t_info.target_bitrate = 0;

    } else {

        t_info.target_bitrate = avc_context->bit_rate;

        t_info.quality        = 0;

    }



    /* Now initialise libtheora */

    h->t_state = th_encode_alloc(&t_info);

    if (!h->t_state) {

        av_log(avc_context, AV_LOG_ERROR, "theora_encode_init failed\n");

        return -1;

    }



    h->keyframe_mask = (1 << t_info.keyframe_granule_shift) - 1;

    /* Clear up theora_info struct */

    th_info_clear(&t_info);



    if (th_encode_ctl(h->t_state, TH_ENCCTL_SET_KEYFRAME_FREQUENCY_FORCE,

                      &gop_size, sizeof(gop_size))) {

        av_log(avc_context, AV_LOG_ERROR, "Error setting GOP size\n");

        return -1;

    }



    // need to enable 2 pass (via TH_ENCCTL_2PASS_) before encoding headers

    if (avc_context->flags & CODEC_FLAG_PASS1) {

        if (get_stats(avc_context, 0))

            return -1;

    } else if (avc_context->flags & CODEC_FLAG_PASS2) {

        if (submit_stats(avc_context))

            return -1;

    }



    /*

        Output first header packet consisting of theora

        header, comment, and tables.



        Each one is prefixed with a 16bit size, then they

        are concatenated together into libavcodec's extradata.

    */

    offset = 0;



    /* Headers */

    th_comment_init(&t_comment);



    while (th_encode_flushheader(h->t_state, &t_comment, &o_packet))

        if (concatenate_packet(&offset, avc_context, &o_packet))

            return -1;



    th_comment_clear(&t_comment);



    /* Set up the output AVFrame */

    avc_context->coded_frame = av_frame_alloc();

    if (!avc_context->coded_frame)

        return AVERROR(ENOMEM);



    return 0;

}

static av_cold int encode_init(AVCodecContext* avc_context)

{

    theora_info t_info;

    theora_comment t_comment;

    ogg_packet o_packet;

    unsigned int offset;

    TheoraContext *h = avc_context->priv_data;



    /* Set up the theora_info struct */

    theora_info_init( &t_info );

    t_info.width = FFALIGN(avc_context->width, 16);

    t_info.height = FFALIGN(avc_context->height, 16);

    t_info.frame_width = avc_context->width;

    t_info.frame_height = avc_context->height;

    t_info.offset_x = 0;

    t_info.offset_y = avc_context->height & 0xf;

    /* Swap numerator and denominator as time_base in AVCodecContext gives the

     * time period between frames, but theora_info needs the framerate.  */

    t_info.fps_numerator = avc_context->time_base.den;

    t_info.fps_denominator = avc_context->time_base.num;

    if (avc_context->sample_aspect_ratio.num != 0) {

        t_info.aspect_numerator = avc_context->sample_aspect_ratio.num;

        t_info.aspect_denominator = avc_context->sample_aspect_ratio.den;

    } else {

        t_info.aspect_numerator = 1;

        t_info.aspect_denominator = 1;

    }

    t_info.colorspace = OC_CS_UNSPECIFIED;

    t_info.pixelformat = OC_PF_420;

    t_info.target_bitrate = avc_context->bit_rate;

    t_info.keyframe_frequency = avc_context->gop_size;

    t_info.keyframe_frequency_force = avc_context->gop_size;

    t_info.keyframe_mindistance = avc_context->keyint_min;

    t_info.quality = 0;



    t_info.quick_p = 1;

    t_info.dropframes_p = 0;

    t_info.keyframe_auto_p = 1;

    t_info.keyframe_data_target_bitrate = t_info.target_bitrate * 1.5;

    t_info.keyframe_auto_threshold = 80;

    t_info.noise_sensitivity = 1;

    t_info.sharpness = 0;



    /* Now initialise libtheora */

    if (theora_encode_init( &(h->t_state), &t_info ) != 0) {

        av_log(avc_context, AV_LOG_ERROR, "theora_encode_init failed\n");

        return -1;

    }



    /* Clear up theora_info struct */

    theora_info_clear( &t_info );



    /*

        Output first header packet consisting of theora

        header, comment, and tables.



        Each one is prefixed with a 16bit size, then they

        are concatenated together into ffmpeg's extradata.

    */

    offset = 0;



    /* Header */

    theora_encode_header( &(h->t_state), &o_packet );

    if (concatenate_packet( &offset, avc_context, &o_packet ) != 0) {

        return -1;

    }



    /* Comment */

    theora_comment_init( &t_comment );

    theora_encode_comment( &t_comment, &o_packet );

    if (concatenate_packet( &offset, avc_context, &o_packet ) != 0) {

        return -1;

    }



    /* Tables */

    theora_encode_tables( &(h->t_state), &o_packet );

    if (concatenate_packet( &offset, avc_context, &o_packet ) != 0) {

        return -1;

    }



    /* Clear up theora_comment struct */

    theora_comment_clear( &t_comment );



    /* Set up the output AVFrame */

    avc_context->coded_frame= avcodec_alloc_frame();



    return 0;

}

static void new_video_stream(AVFormatContext *oc, int file_idx)

{

    AVStream *st;

    OutputStream *ost;

    AVCodecContext *video_enc;

    enum CodecID codec_id = CODEC_ID_NONE;

    AVCodec *codec= NULL;



    if(!video_stream_copy){

        if (video_codec_name) {

            codec_id = find_codec_or_die(video_codec_name, AVMEDIA_TYPE_VIDEO, 1,

                                         avcodec_opts[AVMEDIA_TYPE_VIDEO]->strict_std_compliance);

            codec = avcodec_find_encoder_by_name(video_codec_name);

        } else {

            codec_id = av_guess_codec(oc->oformat, NULL, oc->filename, NULL, AVMEDIA_TYPE_VIDEO);

            codec = avcodec_find_encoder(codec_id);

        }

    }



    ost = new_output_stream(oc, file_idx, codec);

    st  = ost->st;

    if (!video_stream_copy) {

        ost->frame_aspect_ratio = frame_aspect_ratio;

        frame_aspect_ratio = 0;

#if CONFIG_AVFILTER

        ost->avfilter= vfilters;

        vfilters = NULL;

#endif

    }



    ost->bitstream_filters = video_bitstream_filters;

    video_bitstream_filters= NULL;



    st->codec->thread_count= thread_count;



    video_enc = st->codec;



    if(video_codec_tag)

        video_enc->codec_tag= video_codec_tag;



    if(oc->oformat->flags & AVFMT_GLOBALHEADER) {

        video_enc->flags |= CODEC_FLAG_GLOBAL_HEADER;

    }



    if (video_stream_copy) {

        st->stream_copy = 1;

        video_enc->codec_type = AVMEDIA_TYPE_VIDEO;

        video_enc->sample_aspect_ratio =

        st->sample_aspect_ratio = av_d2q(frame_aspect_ratio*frame_height/frame_width, 255);

    } else {

        const char *p;

        int i;



        if (frame_rate.num)

            ost->frame_rate = frame_rate;

        video_enc->codec_id = codec_id;

        set_context_opts(video_enc, avcodec_opts[AVMEDIA_TYPE_VIDEO], AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM, codec);



        video_enc->width = frame_width;

        video_enc->height = frame_height;

        video_enc->pix_fmt = frame_pix_fmt;

        st->sample_aspect_ratio = video_enc->sample_aspect_ratio;



        if (intra_only)

            video_enc->gop_size = 0;

        if (video_qscale || same_quality) {

            video_enc->flags |= CODEC_FLAG_QSCALE;

            video_enc->global_quality = FF_QP2LAMBDA * video_qscale;

        }



        if(intra_matrix)

            video_enc->intra_matrix = intra_matrix;

        if(inter_matrix)

            video_enc->inter_matrix = inter_matrix;



        p= video_rc_override_string;

        for(i=0; p; i++){

            int start, end, q;

            int e=sscanf(p, "%d,%d,%d", &start, &end, &q);

            if(e!=3){

                fprintf(stderr, "error parsing rc_override\n");

                ffmpeg_exit(1);

            }

            video_enc->rc_override=

                av_realloc(video_enc->rc_override,

                           sizeof(RcOverride)*(i+1));

            video_enc->rc_override[i].start_frame= start;

            video_enc->rc_override[i].end_frame  = end;

            if(q>0){

                video_enc->rc_override[i].qscale= q;

                video_enc->rc_override[i].quality_factor= 1.0;

            }

            else{

                video_enc->rc_override[i].qscale= 0;

                video_enc->rc_override[i].quality_factor= -q/100.0;

            }

            p= strchr(p, '/');

            if(p) p++;

        }

        video_enc->rc_override_count=i;

        if (!video_enc->rc_initial_buffer_occupancy)

            video_enc->rc_initial_buffer_occupancy = video_enc->rc_buffer_size*3/4;

        video_enc->me_threshold= me_threshold;

        video_enc->intra_dc_precision= intra_dc_precision - 8;



        if (do_psnr)

            video_enc->flags|= CODEC_FLAG_PSNR;



        /* two pass mode */

        if (do_pass) {

            if (do_pass == 1) {

                video_enc->flags |= CODEC_FLAG_PASS1;

            } else {

                video_enc->flags |= CODEC_FLAG_PASS2;

            }

        }



        if (forced_key_frames)

            parse_forced_key_frames(forced_key_frames, ost, video_enc);

    }

    if (video_language) {

        av_dict_set(&st->metadata, "language", video_language, 0);

        av_freep(&video_language);

    }



    /* reset some key parameters */

    video_disable = 0;

    av_freep(&video_codec_name);

    av_freep(&forced_key_frames);

    video_stream_copy = 0;

    frame_pix_fmt = PIX_FMT_NONE;

}

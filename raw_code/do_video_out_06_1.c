static void do_video_out(AVFormatContext *s,

                         OutputStream *ost,

                         InputStream *ist,

                         AVFrame *in_picture,

                         int *frame_size, float quality)

{

    int nb_frames, i, ret, av_unused resample_changed;

    AVFrame *final_picture, *formatted_picture;

    AVCodecContext *enc, *dec;

    double sync_ipts;



    enc = ost->st->codec;

    dec = ist->st->codec;



    sync_ipts = get_sync_ipts(ost) / av_q2d(enc->time_base);



    /* by default, we output a single frame */

    nb_frames = 1;



    *frame_size = 0;



    if(video_sync_method){

        double vdelta = sync_ipts - ost->sync_opts;

        //FIXME set to 0.5 after we fix some dts/pts bugs like in avidec.c

        if (vdelta < -1.1)

            nb_frames = 0;

        else if (video_sync_method == 2 || (video_sync_method<0 && (s->oformat->flags & AVFMT_VARIABLE_FPS))){

            if(vdelta<=-0.6){

                nb_frames=0;

            }else if(vdelta>0.6)

                ost->sync_opts= lrintf(sync_ipts);

        }else if (vdelta > 1.1)

            nb_frames = lrintf(vdelta);

//fprintf(stderr, "vdelta:%f, ost->sync_opts:%"PRId64", ost->sync_ipts:%f nb_frames:%d\n", vdelta, ost->sync_opts, get_sync_ipts(ost), nb_frames);

        if (nb_frames == 0){

            ++nb_frames_drop;

            if (verbose>2)

                fprintf(stderr, "*** drop!\n");

        }else if (nb_frames > 1) {

            nb_frames_dup += nb_frames - 1;

            if (verbose>2)

                fprintf(stderr, "*** %d dup!\n", nb_frames-1);

        }

    }else

        ost->sync_opts= lrintf(sync_ipts);



    nb_frames= FFMIN(nb_frames, max_frames[AVMEDIA_TYPE_VIDEO] - ost->frame_number);

    if (nb_frames <= 0)

        return;



    formatted_picture = in_picture;

    final_picture = formatted_picture;



#if !CONFIG_AVFILTER

    resample_changed = ost->resample_width   != dec->width  ||

                       ost->resample_height  != dec->height ||

                       ost->resample_pix_fmt != dec->pix_fmt;



    if (resample_changed) {

        av_log(NULL, AV_LOG_INFO,

               "Input stream #%d.%d frame changed from size:%dx%d fmt:%s to size:%dx%d fmt:%s\n",

               ist->file_index, ist->st->index,

               ost->resample_width, ost->resample_height, av_get_pix_fmt_name(ost->resample_pix_fmt),

               dec->width         , dec->height         , av_get_pix_fmt_name(dec->pix_fmt));

        ost->resample_width   = dec->width;

        ost->resample_height  = dec->height;

        ost->resample_pix_fmt = dec->pix_fmt;

    }



    ost->video_resample = dec->width   != enc->width  ||

                          dec->height  != enc->height ||

                          dec->pix_fmt != enc->pix_fmt;



    if (ost->video_resample) {

        final_picture = &ost->resample_frame;

        if (!ost->img_resample_ctx || resample_changed) {

            /* initialize the destination picture */

            if (!ost->resample_frame.data[0]) {

                avcodec_get_frame_defaults(&ost->resample_frame);

                if (avpicture_alloc((AVPicture *)&ost->resample_frame, enc->pix_fmt,

                                    enc->width, enc->height)) {

                    fprintf(stderr, "Cannot allocate temp picture, check pix fmt\n");

                    exit_program(1);

                }

            }

            /* initialize a new scaler context */

            sws_freeContext(ost->img_resample_ctx);

            ost->img_resample_ctx = sws_getContext(dec->width, dec->height, dec->pix_fmt,

                                                   enc->width, enc->height, enc->pix_fmt,

                                                   ost->sws_flags, NULL, NULL, NULL);

            if (ost->img_resample_ctx == NULL) {

                fprintf(stderr, "Cannot get resampling context\n");

                exit_program(1);

            }

        }

        sws_scale(ost->img_resample_ctx, formatted_picture->data, formatted_picture->linesize,

              0, ost->resample_height, final_picture->data, final_picture->linesize);

    }

#else

    if (resample_changed) {

        avfilter_graph_free(&ost->graph);

        if (configure_video_filters(ist, ost)) {

            fprintf(stderr, "Error reinitialising filters!\n");

            exit_program(1);

        }

    }

#endif

    if (resample_changed) {

        ost->resample_width   = dec->width;

        ost->resample_height  = dec->height;

        ost->resample_pix_fmt = dec->pix_fmt;

    }



    /* duplicates frame if needed */

    for(i=0;i<nb_frames;i++) {

        AVPacket pkt;

        av_init_packet(&pkt);

        pkt.stream_index= ost->index;



        if (s->oformat->flags & AVFMT_RAWPICTURE) {

            /* raw pictures are written as AVPicture structure to

               avoid any copies. We support temorarily the older

               method. */

            AVFrame* old_frame = enc->coded_frame;

            enc->coded_frame = dec->coded_frame; //FIXME/XXX remove this hack

            pkt.data= (uint8_t *)final_picture;

            pkt.size=  sizeof(AVPicture);

            pkt.pts= av_rescale_q(ost->sync_opts, enc->time_base, ost->st->time_base);

            pkt.flags |= AV_PKT_FLAG_KEY;



            write_frame(s, &pkt, ost->st->codec, ost->bitstream_filters);

            enc->coded_frame = old_frame;

        } else {

            AVFrame big_picture;



            big_picture= *final_picture;

            /* better than nothing: use input picture interlaced

               settings */

            big_picture.interlaced_frame = in_picture->interlaced_frame;

            if (ost->st->codec->flags & (CODEC_FLAG_INTERLACED_DCT|CODEC_FLAG_INTERLACED_ME)) {

                if(top_field_first == -1)

                    big_picture.top_field_first = in_picture->top_field_first;

                else

                    big_picture.top_field_first = top_field_first;

            }



            /* handles sameq here. This is not correct because it may

               not be a global option */

            big_picture.quality = quality;

            if(!me_threshold)

                big_picture.pict_type = 0;

//            big_picture.pts = AV_NOPTS_VALUE;

            big_picture.pts= ost->sync_opts;

//            big_picture.pts= av_rescale(ost->sync_opts, AV_TIME_BASE*(int64_t)enc->time_base.num, enc->time_base.den);

//av_log(NULL, AV_LOG_DEBUG, "%"PRId64" -> encoder\n", ost->sync_opts);

            if (ost->forced_kf_index < ost->forced_kf_count &&

                big_picture.pts >= ost->forced_kf_pts[ost->forced_kf_index]) {

                big_picture.pict_type = AV_PICTURE_TYPE_I;

                ost->forced_kf_index++;

            }

            ret = avcodec_encode_video(enc,

                                       bit_buffer, bit_buffer_size,

                                       &big_picture);

            if (ret < 0) {

                fprintf(stderr, "Video encoding failed\n");

                exit_program(1);

            }



            if(ret>0){

                pkt.data= bit_buffer;

                pkt.size= ret;

                if(enc->coded_frame->pts != AV_NOPTS_VALUE)

                    pkt.pts= av_rescale_q(enc->coded_frame->pts, enc->time_base, ost->st->time_base);

/*av_log(NULL, AV_LOG_DEBUG, "encoder -> %"PRId64"/%"PRId64"\n",

   pkt.pts != AV_NOPTS_VALUE ? av_rescale(pkt.pts, enc->time_base.den, AV_TIME_BASE*(int64_t)enc->time_base.num) : -1,

   pkt.dts != AV_NOPTS_VALUE ? av_rescale(pkt.dts, enc->time_base.den, AV_TIME_BASE*(int64_t)enc->time_base.num) : -1);*/



                if(enc->coded_frame->key_frame)

                    pkt.flags |= AV_PKT_FLAG_KEY;

                write_frame(s, &pkt, ost->st->codec, ost->bitstream_filters);

                *frame_size = ret;

                video_size += ret;

                //fprintf(stderr,"\nFrame: %3d size: %5d type: %d",

                //        enc->frame_number-1, ret, enc->pict_type);

                /* if two pass, output log */

                if (ost->logfile && enc->stats_out) {

                    fprintf(ost->logfile, "%s", enc->stats_out);

                }

            }

        }

        ost->sync_opts++;

        ost->frame_number++;

    }

}

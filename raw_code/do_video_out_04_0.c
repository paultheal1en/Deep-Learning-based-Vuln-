static void do_video_out(AVFormatContext *s,

                         OutputStream *ost,

                         InputStream *ist,

                         AVFrame *in_picture,

                         int *frame_size, float quality)

{

    int nb_frames, i, ret, format_video_sync;

    AVFrame *final_picture;

    AVCodecContext *enc;

    double sync_ipts;



    enc = ost->st->codec;



    sync_ipts = get_sync_ipts(ost) / av_q2d(enc->time_base);



    /* by default, we output a single frame */

    nb_frames = 1;



    *frame_size = 0;



    format_video_sync = video_sync_method;

    if (format_video_sync < 0)

        format_video_sync = (s->oformat->flags & AVFMT_NOTIMESTAMPS) ? 0 :

                            (s->oformat->flags & AVFMT_VARIABLE_FPS) ? 2 : 1;



    if (format_video_sync) {

        double vdelta = sync_ipts - ost->sync_opts;

        //FIXME set to 0.5 after we fix some dts/pts bugs like in avidec.c

        if (vdelta < -1.1)

            nb_frames = 0;

        else if (format_video_sync == 2) {

            if(vdelta<=-0.6){

                nb_frames=0;

            }else if(vdelta>0.6)

                ost->sync_opts= lrintf(sync_ipts);

        }else if (vdelta > 1.1)

            nb_frames = lrintf(vdelta);

//fprintf(stderr, "vdelta:%f, ost->sync_opts:%"PRId64", ost->sync_ipts:%f nb_frames:%d\n", vdelta, ost->sync_opts, get_sync_ipts(ost), nb_frames);

        if (nb_frames == 0){

            ++nb_frames_drop;

            av_log(NULL, AV_LOG_VERBOSE, "*** drop!\n");

        }else if (nb_frames > 1) {

            nb_frames_dup += nb_frames - 1;

            av_log(NULL, AV_LOG_VERBOSE, "*** %d dup!\n", nb_frames-1);

        }

    }else

        ost->sync_opts= lrintf(sync_ipts);



    nb_frames = FFMIN(nb_frames, ost->max_frames - ost->frame_number);

    if (nb_frames <= 0)

        return;



    do_video_resample(ost, ist, in_picture, &final_picture);



    /* duplicates frame if needed */

    for(i=0;i<nb_frames;i++) {

        AVPacket pkt;

        av_init_packet(&pkt);

        pkt.stream_index= ost->index;



        if (s->oformat->flags & AVFMT_RAWPICTURE) {

            /* raw pictures are written as AVPicture structure to

               avoid any copies. We support temporarily the older

               method. */

            enc->coded_frame->interlaced_frame = in_picture->interlaced_frame;

            enc->coded_frame->top_field_first  = in_picture->top_field_first;

            pkt.data= (uint8_t *)final_picture;

            pkt.size=  sizeof(AVPicture);

            pkt.pts= av_rescale_q(ost->sync_opts, enc->time_base, ost->st->time_base);

            pkt.flags |= AV_PKT_FLAG_KEY;



            write_frame(s, &pkt, ost->st->codec, ost->bitstream_filters);

        } else {

            AVFrame big_picture;



            big_picture= *final_picture;

            /* better than nothing: use input picture interlaced

               settings */

            big_picture.interlaced_frame = in_picture->interlaced_frame;

            if (ost->st->codec->flags & (CODEC_FLAG_INTERLACED_DCT|CODEC_FLAG_INTERLACED_ME)) {

                if (ost->top_field_first == -1)

                    big_picture.top_field_first = in_picture->top_field_first;

                else

                    big_picture.top_field_first = !!ost->top_field_first;

            }



            /* handles same_quant here. This is not correct because it may

               not be a global option */

            big_picture.quality = quality;

            if (!enc->me_threshold)

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

                av_log(NULL, AV_LOG_FATAL, "Video encoding failed\n");

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

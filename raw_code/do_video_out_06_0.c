static void do_video_out(AVFormatContext *s,

                         AVOutputStream *ost,

                         AVInputStream *ist,

                         AVFrame *in_picture,

                         int *frame_size)

{

    int nb_frames, i, ret;

    AVFrame *final_picture, *formatted_picture, *resampling_dst, *padding_src;

    AVFrame picture_crop_temp, picture_pad_temp;

    AVCodecContext *enc, *dec;



    avcodec_get_frame_defaults(&picture_crop_temp);

    avcodec_get_frame_defaults(&picture_pad_temp);



    enc = ost->st->codec;

    dec = ist->st->codec;



    /* by default, we output a single frame */

    nb_frames = 1;



    *frame_size = 0;



    if(video_sync_method>0 || (video_sync_method && av_q2d(enc->time_base) > 0.001)){

        double vdelta;

        vdelta = get_sync_ipts(ost) / av_q2d(enc->time_base) - ost->sync_opts;

        //FIXME set to 0.5 after we fix some dts/pts bugs like in avidec.c

        if (vdelta < -1.1)

            nb_frames = 0;

        else if (video_sync_method == 2)

            ost->sync_opts= lrintf(get_sync_ipts(ost) / av_q2d(enc->time_base));

        else if (vdelta > 1.1)

            nb_frames = lrintf(vdelta);

//fprintf(stderr, "vdelta:%f, ost->sync_opts:%"PRId64", ost->sync_ipts:%f nb_frames:%d\n", vdelta, ost->sync_opts, ost->sync_ipts, nb_frames);

        if (nb_frames == 0){

            ++nb_frames_drop;

            if (verbose>2)

                fprintf(stderr, "*** drop!\n");

        }else if (nb_frames > 1) {

            nb_frames_dup += nb_frames;

            if (verbose>2)

                fprintf(stderr, "*** %d dup!\n", nb_frames-1);

        }

    }else

        ost->sync_opts= lrintf(get_sync_ipts(ost) / av_q2d(enc->time_base));



    nb_frames= FFMIN(nb_frames, max_frames[CODEC_TYPE_VIDEO] - ost->frame_number);

    if (nb_frames <= 0)

        return;



    if (ost->video_crop) {

        if (av_picture_crop((AVPicture *)&picture_crop_temp, (AVPicture *)in_picture, dec->pix_fmt, ost->topBand, ost->leftBand) < 0) {

            av_log(NULL, AV_LOG_ERROR, "error cropping picture\n");

            if (exit_on_error)

                av_exit(1);

            return;

        }

        formatted_picture = &picture_crop_temp;

    } else {

        formatted_picture = in_picture;

    }



    final_picture = formatted_picture;

    padding_src = formatted_picture;

    resampling_dst = &ost->pict_tmp;

    if (ost->video_pad) {

        final_picture = &ost->pict_tmp;

        if (ost->video_resample) {

            if (av_picture_crop((AVPicture *)&picture_pad_temp, (AVPicture *)final_picture, enc->pix_fmt, ost->padtop, ost->padleft) < 0) {

                av_log(NULL, AV_LOG_ERROR, "error padding picture\n");

                if (exit_on_error)

                    av_exit(1);

                return;

            }

            resampling_dst = &picture_pad_temp;

        }

    }



    if (ost->video_resample) {

        padding_src = NULL;

        final_picture = &ost->pict_tmp;

        sws_scale(ost->img_resample_ctx, formatted_picture->data, formatted_picture->linesize,

              0, ost->resample_height, resampling_dst->data, resampling_dst->linesize);

    }



    if (ost->video_pad) {

        av_picture_pad((AVPicture*)final_picture, (AVPicture *)padding_src,

                enc->height, enc->width, enc->pix_fmt,

                ost->padtop, ost->padbottom, ost->padleft, ost->padright, padcolor);

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

            pkt.flags |= PKT_FLAG_KEY;



            write_frame(s, &pkt, ost->st->codec, bitstream_filters[ost->file_index][pkt.stream_index]);

            enc->coded_frame = old_frame;

        } else {

            AVFrame big_picture;



            big_picture= *final_picture;

            /* better than nothing: use input picture interlaced

               settings */

            big_picture.interlaced_frame = in_picture->interlaced_frame;

            if(avctx_opts[CODEC_TYPE_VIDEO]->flags & (CODEC_FLAG_INTERLACED_DCT|CODEC_FLAG_INTERLACED_ME)){

                if(top_field_first == -1)

                    big_picture.top_field_first = in_picture->top_field_first;

                else

                    big_picture.top_field_first = top_field_first;

            }



            /* handles sameq here. This is not correct because it may

               not be a global option */

            if (same_quality) {

                big_picture.quality = ist->st->quality;

            }else

                big_picture.quality = ost->st->quality;

            if(!me_threshold)

                big_picture.pict_type = 0;

//            big_picture.pts = AV_NOPTS_VALUE;

            big_picture.pts= ost->sync_opts;

//            big_picture.pts= av_rescale(ost->sync_opts, AV_TIME_BASE*(int64_t)enc->time_base.num, enc->time_base.den);

//av_log(NULL, AV_LOG_DEBUG, "%"PRId64" -> encoder\n", ost->sync_opts);

            ret = avcodec_encode_video(enc,

                                       bit_buffer, bit_buffer_size,

                                       &big_picture);

            if (ret == -1) {

                fprintf(stderr, "Video encoding failed\n");

                av_exit(1);

            }

            //enc->frame_number = enc->real_pict_num;

            if(ret>0){

                pkt.data= bit_buffer;

                pkt.size= ret;

                if(enc->coded_frame->pts != AV_NOPTS_VALUE)

                    pkt.pts= av_rescale_q(enc->coded_frame->pts, enc->time_base, ost->st->time_base);

/*av_log(NULL, AV_LOG_DEBUG, "encoder -> %"PRId64"/%"PRId64"\n",

   pkt.pts != AV_NOPTS_VALUE ? av_rescale(pkt.pts, enc->time_base.den, AV_TIME_BASE*(int64_t)enc->time_base.num) : -1,

   pkt.dts != AV_NOPTS_VALUE ? av_rescale(pkt.dts, enc->time_base.den, AV_TIME_BASE*(int64_t)enc->time_base.num) : -1);*/



                if(enc->coded_frame->key_frame)

                    pkt.flags |= PKT_FLAG_KEY;

                write_frame(s, &pkt, ost->st->codec, bitstream_filters[ost->file_index][pkt.stream_index]);

                *frame_size = ret;

                video_size += ret;

                //fprintf(stderr,"\nFrame: %3d %3d size: %5d type: %d",

                //        enc->frame_number-1, enc->real_pict_num, ret,

                //        enc->pict_type);

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

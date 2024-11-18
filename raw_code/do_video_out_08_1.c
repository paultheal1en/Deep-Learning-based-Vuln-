static void do_video_out(AVFormatContext *s,

                         AVOutputStream *ost,

                         AVInputStream *ist,

                         AVFrame *in_picture,

                         int *frame_size)

{

    int nb_frames, i, ret;

    int64_t topBand, bottomBand, leftBand, rightBand;

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



    if(video_sync_method){

        double vdelta;

        vdelta = get_sync_ipts(ost) / av_q2d(enc->time_base) - ost->sync_opts;

        //FIXME set to 0.5 after we fix some dts/pts bugs like in avidec.c

        if (vdelta < -1.1)

            nb_frames = 0;

        else if (video_sync_method == 2 || (video_sync_method<0 && (s->oformat->flags & AVFMT_VARIABLE_FPS))){

            if(vdelta<=-0.6){

                nb_frames=0;

            }else if(vdelta>0.6)

            ost->sync_opts= lrintf(get_sync_ipts(ost) / av_q2d(enc->time_base));

        }else if (vdelta > 1.1)

            nb_frames = lrintf(vdelta);

//fprintf(stderr, "vdelta:%f, ost->sync_opts:%"PRId64", ost->sync_ipts:%f nb_frames:%d\n", vdelta, ost->sync_opts, get_sync_ipts(ost), nb_frames);

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

            fprintf(stderr, "error cropping picture\n");

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

                fprintf(stderr, "error padding picture\n");

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

        if(  (ost->resample_height != (ist->st->codec->height - (ost->topBand  + ost->bottomBand)))

          || (ost->resample_width  != (ist->st->codec->width  - (ost->leftBand + ost->rightBand)))

          || (ost->resample_pix_fmt!= ist->st->codec->pix_fmt) ) {



            fprintf(stderr,"Input Stream #%d.%d frame size changed to %dx%d, %s\n", ist->file_index, ist->index, ist->st->codec->width, ist->st->codec->height,avcodec_get_pix_fmt_name(ist->st->codec->pix_fmt));

            /* keep bands proportional to the frame size */

            topBand    = ((int64_t)ist->st->codec->height * ost->original_topBand    / ost->original_height) & ~1;

            bottomBand = ((int64_t)ist->st->codec->height * ost->original_bottomBand / ost->original_height) & ~1;

            leftBand   = ((int64_t)ist->st->codec->width  * ost->original_leftBand   / ost->original_width)  & ~1;

            rightBand  = ((int64_t)ist->st->codec->width  * ost->original_rightBand  / ost->original_width)  & ~1;



            /* sanity check to ensure no bad band sizes sneak in */

            assert(topBand    <= INT_MAX && topBand    >= 0);

            assert(bottomBand <= INT_MAX && bottomBand >= 0);

            assert(leftBand   <= INT_MAX && leftBand   >= 0);

            assert(rightBand  <= INT_MAX && rightBand  >= 0);



            ost->topBand    = topBand;

            ost->bottomBand = bottomBand;

            ost->leftBand   = leftBand;

            ost->rightBand  = rightBand;



            ost->resample_height = ist->st->codec->height - (ost->topBand  + ost->bottomBand);

            ost->resample_width  = ist->st->codec->width  - (ost->leftBand + ost->rightBand);

            ost->resample_pix_fmt= ist->st->codec->pix_fmt;



            /* initialize a new scaler context */

            sws_freeContext(ost->img_resample_ctx);

            sws_flags = av_get_int(sws_opts, "sws_flags", NULL);

            ost->img_resample_ctx = sws_getContext(

                ist->st->codec->width  - (ost->leftBand + ost->rightBand),

                ist->st->codec->height - (ost->topBand  + ost->bottomBand),

                ist->st->codec->pix_fmt,

                ost->st->codec->width  - (ost->padleft  + ost->padright),

                ost->st->codec->height - (ost->padtop   + ost->padbottom),

                ost->st->codec->pix_fmt,

                sws_flags, NULL, NULL, NULL);

            if (ost->img_resample_ctx == NULL) {

                fprintf(stderr, "Cannot get resampling context\n");

                av_exit(1);

            }

        }

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

            if(avcodec_opts[CODEC_TYPE_VIDEO]->flags & (CODEC_FLAG_INTERLACED_DCT|CODEC_FLAG_INTERLACED_ME)){

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

            if (ret < 0) {

                fprintf(stderr, "Video encoding failed\n");

                av_exit(1);

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

                    pkt.flags |= PKT_FLAG_KEY;

                write_frame(s, &pkt, ost->st->codec, bitstream_filters[ost->file_index][pkt.stream_index]);

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

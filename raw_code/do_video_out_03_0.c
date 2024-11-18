static void do_video_out(AVFormatContext *s, 

                         AVOutputStream *ost, 

                         AVInputStream *ist,

                         AVFrame *in_picture,

                         int *frame_size, AVOutputStream *audio_sync)

{

    int nb_frames, i, ret;

    AVFrame *final_picture, *formatted_picture;

    AVFrame picture_format_temp, picture_crop_temp;

    static uint8_t *video_buffer= NULL;

    uint8_t *buf = NULL, *buf1 = NULL;

    AVCodecContext *enc, *dec;

    enum PixelFormat target_pixfmt;

    

#define VIDEO_BUFFER_SIZE (1024*1024)



    avcodec_get_frame_defaults(&picture_format_temp);

    avcodec_get_frame_defaults(&picture_crop_temp);



    enc = &ost->st->codec;

    dec = &ist->st->codec;



    /* by default, we output a single frame */

    nb_frames = 1;



    *frame_size = 0;



    if(sync_method){

        double vdelta;

        vdelta = ost->sync_ipts * enc->frame_rate / enc->frame_rate_base - ost->sync_opts;

        //FIXME set to 0.5 after we fix some dts/pts bugs like in avidec.c

        if (vdelta < -1.1)

            nb_frames = 0;

        else if (vdelta > 1.1)

            nb_frames = 2;

//printf("vdelta:%f, ost->sync_opts:%lld, ost->sync_ipts:%f nb_frames:%d\n", vdelta, ost->sync_opts, ost->sync_ipts, nb_frames);

    }

    ost->sync_opts+= nb_frames;



    if (nb_frames <= 0) 

        return;



    if (!video_buffer)

        video_buffer = av_malloc(VIDEO_BUFFER_SIZE);

    if (!video_buffer)

        return;



    /* convert pixel format if needed */

    target_pixfmt = ost->video_resample || ost->video_pad

        ? PIX_FMT_YUV420P : enc->pix_fmt;

    if (dec->pix_fmt != target_pixfmt) {

        int size;



        /* create temporary picture */

        size = avpicture_get_size(target_pixfmt, dec->width, dec->height);

        buf = av_malloc(size);

        if (!buf)

            return;

        formatted_picture = &picture_format_temp;

        avpicture_fill((AVPicture*)formatted_picture, buf, target_pixfmt, dec->width, dec->height);

        

        if (img_convert((AVPicture*)formatted_picture, target_pixfmt, 

                        (AVPicture *)in_picture, dec->pix_fmt, 

                        dec->width, dec->height) < 0) {



            if (verbose >= 0)

                fprintf(stderr, "pixel format conversion not handled\n");



            goto the_end;

        }

    } else {

        formatted_picture = in_picture;

    }



    /* XXX: resampling could be done before raw format conversion in

       some cases to go faster */

    /* XXX: only works for YUV420P */

    if (ost->video_resample) {

        final_picture = &ost->pict_tmp;

        img_resample(ost->img_resample_ctx, (AVPicture*)final_picture, (AVPicture*)formatted_picture);

       

        if (ost->padtop || ost->padbottom || ost->padleft || ost->padright) {

            fill_pad_region((AVPicture*)final_picture, enc->height, enc->width,

                    ost->padtop, ost->padbottom, ost->padleft, ost->padright,

                    padcolor);

        }

        

	if (enc->pix_fmt != PIX_FMT_YUV420P) {

            int size;

	    

	    av_free(buf);

            /* create temporary picture */

            size = avpicture_get_size(enc->pix_fmt, enc->width, enc->height);

            buf = av_malloc(size);

            if (!buf)

                return;

            final_picture = &picture_format_temp;

            avpicture_fill((AVPicture*)final_picture, buf, enc->pix_fmt, enc->width, enc->height);

        

            if (img_convert((AVPicture*)final_picture, enc->pix_fmt, 

                            (AVPicture*)&ost->pict_tmp, PIX_FMT_YUV420P, 

                            enc->width, enc->height) < 0) {



                if (verbose >= 0)

                    fprintf(stderr, "pixel format conversion not handled\n");



                goto the_end;

            }

	}

    } else if (ost->video_crop) {

        picture_crop_temp.data[0] = formatted_picture->data[0] +

                (ost->topBand * formatted_picture->linesize[0]) + ost->leftBand;



        picture_crop_temp.data[1] = formatted_picture->data[1] +

                ((ost->topBand >> 1) * formatted_picture->linesize[1]) +

                (ost->leftBand >> 1);



        picture_crop_temp.data[2] = formatted_picture->data[2] +

                ((ost->topBand >> 1) * formatted_picture->linesize[2]) +

                (ost->leftBand >> 1);



        picture_crop_temp.linesize[0] = formatted_picture->linesize[0];

        picture_crop_temp.linesize[1] = formatted_picture->linesize[1];

        picture_crop_temp.linesize[2] = formatted_picture->linesize[2];

        final_picture = &picture_crop_temp;

    } else if (ost->video_pad) {

        final_picture = &ost->pict_tmp;



        for (i = 0; i < 3; i++) {

            uint8_t *optr, *iptr;

            int shift = (i == 0) ? 0 : 1;

            int y, yheight;

            

            /* set offset to start writing image into */

            optr = final_picture->data[i] + (((final_picture->linesize[i] * 

                            ost->padtop) + ost->padleft) >> shift);

            iptr = formatted_picture->data[i];



            yheight = (enc->height - ost->padtop - ost->padbottom) >> shift;

            for (y = 0; y < yheight; y++) {

                /* copy unpadded image row into padded image row */

                memcpy(optr, iptr, formatted_picture->linesize[i]);

                optr += final_picture->linesize[i];

                iptr += formatted_picture->linesize[i];

            }

        }



        fill_pad_region((AVPicture*)final_picture, enc->height, enc->width,

                ost->padtop, ost->padbottom, ost->padleft, ost->padright,

                padcolor);

        

        if (enc->pix_fmt != PIX_FMT_YUV420P) {

            int size;



            av_free(buf);

            /* create temporary picture */

            size = avpicture_get_size(enc->pix_fmt, enc->width, enc->height);

            buf = av_malloc(size);

            if (!buf)

                return;

            final_picture = &picture_format_temp;

            avpicture_fill((AVPicture*)final_picture, buf, enc->pix_fmt, enc->width, enc->height);



            if (img_convert((AVPicture*)final_picture, enc->pix_fmt, 

                        (AVPicture*)&ost->pict_tmp, PIX_FMT_YUV420P, 

                        enc->width, enc->height) < 0) {



                if (verbose >= 0)

                    fprintf(stderr, "pixel format conversion not handled\n");



                goto the_end;

            }

        }

    } else {

        final_picture = formatted_picture;

    }

    /* duplicates frame if needed */

    /* XXX: pb because no interleaving */

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

            if(dec->coded_frame)

                pkt.pts= dec->coded_frame->pts;

            if(dec->coded_frame && dec->coded_frame->key_frame)

                pkt.flags |= PKT_FLAG_KEY;



            av_write_frame(s, &pkt);

	    enc->coded_frame = old_frame;

        } else {

            AVFrame big_picture;



            big_picture= *final_picture;

            /* better than nothing: use input picture interlaced

               settings */

            big_picture.interlaced_frame = in_picture->interlaced_frame;

            if(do_interlace_me || do_interlace_dct){

                if(top_field_first == -1)

                    big_picture.top_field_first = in_picture->top_field_first;

                else

                    big_picture.top_field_first = 1;

            }



            /* handles sameq here. This is not correct because it may

               not be a global option */

            if (same_quality) {

                big_picture.quality = ist->st->quality;

            }else

                big_picture.quality = ost->st->quality;

            if(!me_threshold)

                big_picture.pict_type = 0;

            big_picture.pts = AV_NOPTS_VALUE; //FIXME

            ret = avcodec_encode_video(enc, 

                                       video_buffer, VIDEO_BUFFER_SIZE,

                                       &big_picture);

            //enc->frame_number = enc->real_pict_num;

            if(ret){

                pkt.data= video_buffer;

                pkt.size= ret;

                if(enc->coded_frame)

                    pkt.pts= enc->coded_frame->pts;

                if(enc->coded_frame && enc->coded_frame->key_frame)

                    pkt.flags |= PKT_FLAG_KEY;

                av_write_frame(s, &pkt);

                *frame_size = ret;

                //fprintf(stderr,"\nFrame: %3d %3d size: %5d type: %d",

                //        enc->frame_number-1, enc->real_pict_num, ret,

                //        enc->pict_type);

                /* if two pass, output log */

                if (ost->logfile && enc->stats_out) {

                    fprintf(ost->logfile, "%s", enc->stats_out);

                }

            }

        }

        ost->frame_number++;

    }

 the_end:

    av_free(buf);

    av_free(buf1);

}

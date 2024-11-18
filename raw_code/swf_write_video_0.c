static int swf_write_video(AVFormatContext *s, 

                           AVCodecContext *enc, const uint8_t *buf, int size)

{

    SWFContext *swf = s->priv_data;

    ByteIOContext *pb = &s->pb;

    int c = 0;

    int outSize = 0;

    int outSamples = 0;

    

    /* Flash Player limit */

    if ( swf->swf_frame_number >= 16000 ) {

        return 0;

    }



    /* Store video data in queue */

    if ( enc->codec_type == CODEC_TYPE_VIDEO ) {

        SWFFrame *new_frame = av_malloc(sizeof(SWFFrame));

        new_frame->prev = 0;

        new_frame->next = swf->frame_head;

        new_frame->data = av_malloc(size);

        new_frame->size = size;

        memcpy(new_frame->data,buf,size);

        swf->frame_head = new_frame;

        if ( swf->frame_tail == 0 ) {

            swf->frame_tail = new_frame;

        }

    }

    

    if ( swf->audio_type ) {

        /* Prescan audio data for this swf frame */

retry_swf_audio_packet:

        if ( ( swf->audio_size-outSize ) >= 4 ) {

            int mp3FrameSize = 0;

            int mp3SampleRate = 0;

            int mp3IsMono = 0;

            int mp3SamplesPerFrame = 0;

            

            /* copy out mp3 header from ring buffer */

            uint8_t header[4];

            for (c=0; c<4; c++) {

                header[c] = swf->audio_fifo[(swf->audio_in_pos+outSize+c) % AUDIO_FIFO_SIZE];

            }

            

            if ( swf_mp3_info(header,&mp3FrameSize,&mp3SamplesPerFrame,&mp3SampleRate,&mp3IsMono) ) {

                if ( ( swf->audio_size-outSize ) >= mp3FrameSize ) {

                    outSize += mp3FrameSize;

                    outSamples += mp3SamplesPerFrame;

                    if ( ( swf->sound_samples + outSamples + swf->samples_per_frame ) < swf->video_samples ) {

                        goto retry_swf_audio_packet;

                    }

                }

            } else {

                /* invalid mp3 data, skip forward

                we need to do this since the Flash Player 

                does not like custom headers */

                swf->audio_in_pos ++;

                swf->audio_size --;

                swf->audio_in_pos %= AUDIO_FIFO_SIZE;

                goto retry_swf_audio_packet;

            }

        }

        

        /* audio stream is behind video stream, bail */

        if ( ( swf->sound_samples + outSamples + swf->samples_per_frame ) < swf->video_samples ) {

            return 0;

        }



        /* compute audio/video drift */

        if ( enc->codec_type == CODEC_TYPE_VIDEO ) {

            swf->skip_samples = (int)( ( (double)(swf->swf_frame_number) * (double)enc->frame_rate_base * 44100. ) / (double)(enc->frame_rate) );

            swf->skip_samples -=  swf->video_samples;

        }

    }



    /* check if we need to insert a padding frame */

    if (swf->skip_samples <= ( swf->samples_per_frame / 2 ) ) {

        /* no, it is time for a real frame, check if one is available */

        if ( swf->frame_tail ) {

            if ( swf->video_type == CODEC_ID_FLV1 ) {

                if ( swf->video_frame_number == 0 ) {

                    /* create a new video object */

                    put_swf_tag(s, TAG_VIDEOSTREAM);

                    put_le16(pb, VIDEO_ID);

                    put_le16(pb, 15000 ); /* hard flash player limit */

                    put_le16(pb, enc->width);

                    put_le16(pb, enc->height);

                    put_byte(pb, 0);

                    put_byte(pb, SWF_VIDEO_CODEC_FLV1);

                    put_swf_end_tag(s);

                    

                    /* place the video object for the first time */

                    put_swf_tag(s, TAG_PLACEOBJECT2);

                    put_byte(pb, 0x36);

                    put_le16(pb, 1);

                    put_le16(pb, VIDEO_ID);

                    put_swf_matrix(pb, 1 << FRAC_BITS, 0, 0, 1 << FRAC_BITS, 0, 0);

                    put_le16(pb, swf->video_frame_number );

                    put_byte(pb, 'v');

                    put_byte(pb, 'i');

                    put_byte(pb, 'd');

                    put_byte(pb, 'e');

                    put_byte(pb, 'o');

                    put_byte(pb, 0x00);

                    put_swf_end_tag(s);

                } else {

                    /* mark the character for update */

                    put_swf_tag(s, TAG_PLACEOBJECT2);

                    put_byte(pb, 0x11);

                    put_le16(pb, 1);

                    put_le16(pb, swf->video_frame_number );

                    put_swf_end_tag(s);

                }

    

                // write out pending frames

                for (; ( enc->frame_number - swf->video_frame_number ) > 0;) {

                    /* set video frame data */

                    put_swf_tag(s, TAG_VIDEOFRAME | TAG_LONG);

                    put_le16(pb, VIDEO_ID); 

                    put_le16(pb, swf->video_frame_number++ );

                    put_buffer(pb, swf->frame_tail->data, swf->frame_tail->size);

                    put_swf_end_tag(s);

                }



            } else if ( swf->video_type == CODEC_ID_MJPEG ) {

                if (swf->swf_frame_number > 0) {

                    /* remove the shape */

                    put_swf_tag(s, TAG_REMOVEOBJECT);

                    put_le16(pb, SHAPE_ID); /* shape ID */

                    put_le16(pb, 1); /* depth */

                    put_swf_end_tag(s);

                

                    /* free the bitmap */

                    put_swf_tag(s, TAG_FREECHARACTER);

                    put_le16(pb, BITMAP_ID);

                    put_swf_end_tag(s);

                }

        

                put_swf_tag(s, TAG_JPEG2 | TAG_LONG);

        

                put_le16(pb, BITMAP_ID); /* ID of the image */

        

                /* a dummy jpeg header seems to be required */

                put_byte(pb, 0xff); 

                put_byte(pb, 0xd8);

                put_byte(pb, 0xff);

                put_byte(pb, 0xd9);

                /* write the jpeg image */

                put_buffer(pb, swf->frame_tail->data, swf->frame_tail->size);

        

                put_swf_end_tag(s);

        

                /* draw the shape */

        

                put_swf_tag(s, TAG_PLACEOBJECT);

                put_le16(pb, SHAPE_ID); /* shape ID */

                put_le16(pb, 1); /* depth */

                put_swf_matrix(pb, 20 << FRAC_BITS, 0, 0, 20 << FRAC_BITS, 0, 0);

                put_swf_end_tag(s);

            } else {

                /* invalid codec */

            }

    

            av_free(swf->frame_tail->data);

            swf->frame_tail = swf->frame_tail->prev;

            if ( swf->frame_tail ) {

                if ( swf->frame_tail->next ) {

                    av_free(swf->frame_tail->next);

                }

                swf->frame_tail->next = 0;

            } else {

                swf->frame_head = 0;

            }

            swf->swf_frame_number ++;

        }

    }



    swf->video_samples += swf->samples_per_frame;



    /* streaming sound always should be placed just before showframe tags */

    if ( outSize > 0 ) {

        put_swf_tag(s, TAG_STREAMBLOCK | TAG_LONG);

        put_le16(pb, outSamples);

        put_le16(pb, 0);

        for (c=0; c<outSize; c++) {

            put_byte(pb,swf->audio_fifo[(swf->audio_in_pos+c) % AUDIO_FIFO_SIZE]);

        }

        put_swf_end_tag(s);

    

        /* update FIFO */

        swf->sound_samples += outSamples;

        swf->audio_in_pos += outSize;

        swf->audio_size -= outSize;

        swf->audio_in_pos %= AUDIO_FIFO_SIZE;

    }



    /* output the frame */

    put_swf_tag(s, TAG_SHOWFRAME);

    put_swf_end_tag(s);

    

    put_flush_packet(&s->pb);

    

    return 0;

}

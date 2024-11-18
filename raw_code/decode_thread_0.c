static int decode_thread(void *arg)

{

    VideoState *is = arg;

    AVFormatContext *ic;

    int err, i, ret, video_index, audio_index, subtitle_index;

    AVPacket pkt1, *pkt = &pkt1;

    AVFormatParameters params, *ap = &params;



    video_index = -1;

    audio_index = -1;

    subtitle_index = -1;

    is->video_stream = -1;

    is->audio_stream = -1;

    is->subtitle_stream = -1;



    global_video_state = is;

    url_set_interrupt_cb(decode_interrupt_cb);



    memset(ap, 0, sizeof(*ap));



    ap->width = frame_width;

    ap->height= frame_height;

    ap->time_base= (AVRational){1, 25};

    ap->pix_fmt = frame_pix_fmt;



    err = av_open_input_file(&ic, is->filename, is->iformat, 0, ap);

    if (err < 0) {

        print_error(is->filename, err);

        ret = -1;

        goto fail;

    }

    is->ic = ic;



    if(genpts)

        ic->flags |= AVFMT_FLAG_GENPTS;



    err = av_find_stream_info(ic);

    if (err < 0) {

        fprintf(stderr, "%s: could not find codec parameters\n", is->filename);

        ret = -1;

        goto fail;

    }

    if(ic->pb)

        ic->pb->eof_reached= 0; //FIXME hack, ffplay maybe should not use url_feof() to test for the end



    /* if seeking requested, we execute it */

    if (start_time != AV_NOPTS_VALUE) {

        int64_t timestamp;



        timestamp = start_time;

        /* add the stream start time */

        if (ic->start_time != AV_NOPTS_VALUE)

            timestamp += ic->start_time;

        ret = av_seek_frame(ic, -1, timestamp, AVSEEK_FLAG_BACKWARD);

        if (ret < 0) {

            fprintf(stderr, "%s: could not seek to position %0.3f\n",

                    is->filename, (double)timestamp / AV_TIME_BASE);

        }

    }



    for(i = 0; i < ic->nb_streams; i++) {

        AVCodecContext *enc = ic->streams[i]->codec;

        ic->streams[i]->discard = AVDISCARD_ALL;

        switch(enc->codec_type) {

        case CODEC_TYPE_AUDIO:

            if ((audio_index < 0 || wanted_audio_stream-- > 0) && !audio_disable)

                audio_index = i;

            break;

        case CODEC_TYPE_VIDEO:

            if ((video_index < 0 || wanted_video_stream-- > 0) && !video_disable)

                video_index = i;

            break;

        case CODEC_TYPE_SUBTITLE:

            if (wanted_subtitle_stream-- >= 0 && !video_disable)

                subtitle_index = i;

            break;

        default:

            break;

        }

    }

    if (show_status) {

        dump_format(ic, 0, is->filename, 0);

        dump_stream_info(ic);

    }



    /* open the streams */

    if (audio_index >= 0) {

        stream_component_open(is, audio_index);

    }



    if (video_index >= 0) {

        stream_component_open(is, video_index);

    } else {

        if (!display_disable)

            is->show_audio = 1;

    }



    if (subtitle_index >= 0) {

        stream_component_open(is, subtitle_index);

    }



    if (is->video_stream < 0 && is->audio_stream < 0) {

        fprintf(stderr, "%s: could not open codecs\n", is->filename);

        ret = -1;

        goto fail;

    }



    for(;;) {

        if (is->abort_request)

            break;

        if (is->paused != is->last_paused) {

            is->last_paused = is->paused;

            if (is->paused)

                av_read_pause(ic);

            else

                av_read_play(ic);

        }

#if CONFIG_RTSP_DEMUXER

        if (is->paused && !strcmp(ic->iformat->name, "rtsp")) {

            /* wait 10 ms to avoid trying to get another packet */

            /* XXX: horrible */

            SDL_Delay(10);

            continue;

        }

#endif

        if (is->seek_req) {

            int stream_index= -1;

            int64_t seek_target= is->seek_pos;



            if     (is->   video_stream >= 0) stream_index= is->   video_stream;

            else if(is->   audio_stream >= 0) stream_index= is->   audio_stream;

            else if(is->subtitle_stream >= 0) stream_index= is->subtitle_stream;



            if(stream_index>=0){

                seek_target= av_rescale_q(seek_target, AV_TIME_BASE_Q, ic->streams[stream_index]->time_base);

            }



            ret = av_seek_frame(is->ic, stream_index, seek_target, is->seek_flags);

            if (ret < 0) {

                fprintf(stderr, "%s: error while seeking\n", is->ic->filename);

            }else{

                if (is->audio_stream >= 0) {

                    packet_queue_flush(&is->audioq);

                    packet_queue_put(&is->audioq, &flush_pkt);

                }

                if (is->subtitle_stream >= 0) {

                    packet_queue_flush(&is->subtitleq);

                    packet_queue_put(&is->subtitleq, &flush_pkt);

                }

                if (is->video_stream >= 0) {

                    packet_queue_flush(&is->videoq);

                    packet_queue_put(&is->videoq, &flush_pkt);

                }

            }

            is->seek_req = 0;

        }



        /* if the queue are full, no need to read more */

        if (is->audioq.size > MAX_AUDIOQ_SIZE ||

            is->videoq.size > MAX_VIDEOQ_SIZE ||

            is->subtitleq.size > MAX_SUBTITLEQ_SIZE) {

            /* wait 10 ms */

            SDL_Delay(10);

            continue;

        }

        if(url_feof(ic->pb)) {

            av_init_packet(pkt);

            pkt->data=NULL;

            pkt->size=0;

            pkt->stream_index= is->video_stream;

            packet_queue_put(&is->videoq, pkt);

            continue;

        }

        ret = av_read_frame(ic, pkt);

        if (ret < 0) {

            if (ret != AVERROR_EOF && url_ferror(ic->pb) == 0) {

                SDL_Delay(100); /* wait for user event */

                continue;

            } else

                break;

        }

        if (pkt->stream_index == is->audio_stream) {

            packet_queue_put(&is->audioq, pkt);

        } else if (pkt->stream_index == is->video_stream) {

            packet_queue_put(&is->videoq, pkt);

        } else if (pkt->stream_index == is->subtitle_stream) {

            packet_queue_put(&is->subtitleq, pkt);

        } else {

            av_free_packet(pkt);

        }

    }

    /* wait until the end */

    while (!is->abort_request) {

        SDL_Delay(100);

    }



    ret = 0;

 fail:

    /* disable interrupting */

    global_video_state = NULL;



    /* close each stream */

    if (is->audio_stream >= 0)

        stream_component_close(is, is->audio_stream);

    if (is->video_stream >= 0)

        stream_component_close(is, is->video_stream);

    if (is->subtitle_stream >= 0)

        stream_component_close(is, is->subtitle_stream);

    if (is->ic) {

        av_close_input_file(is->ic);

        is->ic = NULL; /* safety */

    }

    url_set_interrupt_cb(NULL);



    if (ret != 0) {

        SDL_Event event;



        event.type = FF_QUIT_EVENT;

        event.user.data1 = is;

        SDL_PushEvent(&event);

    }

    return 0;

}

int av_find_stream_info(AVFormatContext *ic)

{

    int i, count, ret, read_size, j;

    AVStream *st;

    AVPacket pkt1, *pkt;

    int64_t old_offset = url_ftell(ic->pb);

    struct {

        int64_t last_dts;

        int64_t duration_gcd;

        int duration_count;

        double duration_error[MAX_STD_TIMEBASES];

        int64_t codec_info_duration;

    } info[MAX_STREAMS] = {{0}};



    for(i=0;i<ic->nb_streams;i++) {

        st = ic->streams[i];

        if (st->codec->codec_id == CODEC_ID_AAC) {

            st->codec->sample_rate = 0;

            st->codec->frame_size = 0;

            st->codec->channels = 0;

        }

        if(st->codec->codec_type == AVMEDIA_TYPE_VIDEO){

/*            if(!st->time_base.num)

                st->time_base= */

            if(!st->codec->time_base.num)

                st->codec->time_base= st->time_base;

        }

        //only for the split stuff

        if (!st->parser && !(ic->flags & AVFMT_FLAG_NOPARSE)) {

            st->parser = av_parser_init(st->codec->codec_id);

            if(st->need_parsing == AVSTREAM_PARSE_HEADERS && st->parser){

                st->parser->flags |= PARSER_FLAG_COMPLETE_FRAMES;

            }

        }

        assert(!st->codec->codec);

        //try to just open decoders, in case this is enough to get parameters

        if(!has_codec_parameters(st->codec)){

            AVCodec *codec = avcodec_find_decoder(st->codec->codec_id);

            if (codec)

                avcodec_open(st->codec, codec);

        }

    }



    for(i=0;i<MAX_STREAMS;i++){

        info[i].last_dts= AV_NOPTS_VALUE;

    }



    count = 0;

    read_size = 0;

    for(;;) {

        if(url_interrupt_cb()){

            ret= AVERROR(EINTR);

            av_log(ic, AV_LOG_DEBUG, "interrupted\n");

            break;

        }



        /* check if one codec still needs to be handled */

        for(i=0;i<ic->nb_streams;i++) {

            st = ic->streams[i];

            if (!has_codec_parameters(st->codec))

                break;

            /* variable fps and no guess at the real fps */

            if(   tb_unreliable(st->codec) && !(st->r_frame_rate.num && st->avg_frame_rate.num)

               && info[i].duration_count<20 && st->codec->codec_type == AVMEDIA_TYPE_VIDEO)

                break;

            if(st->parser && st->parser->parser->split && !st->codec->extradata)

                break;

            if(st->first_dts == AV_NOPTS_VALUE)

                break;

        }

        if (i == ic->nb_streams) {

            /* NOTE: if the format has no header, then we need to read

               some packets to get most of the streams, so we cannot

               stop here */

            if (!(ic->ctx_flags & AVFMTCTX_NOHEADER)) {

                /* if we found the info for all the codecs, we can stop */

                ret = count;

                av_log(ic, AV_LOG_DEBUG, "All info found\n");

                break;

            }

        }

        /* we did not get all the codec info, but we read too much data */

        if (read_size >= ic->probesize) {

            ret = count;

            av_log(ic, AV_LOG_DEBUG, "Probe buffer size limit %d reached\n", ic->probesize);

            break;

        }



        /* NOTE: a new stream can be added there if no header in file

           (AVFMTCTX_NOHEADER) */

        ret = av_read_frame_internal(ic, &pkt1);

        if(ret == AVERROR(EAGAIN))

            continue;

        if (ret < 0) {

            /* EOF or error */

            ret = -1; /* we could not have all the codec parameters before EOF */

            for(i=0;i<ic->nb_streams;i++) {

                st = ic->streams[i];

                if (!has_codec_parameters(st->codec)){

                    char buf[256];

                    avcodec_string(buf, sizeof(buf), st->codec, 0);

                    av_log(ic, AV_LOG_WARNING, "Could not find codec parameters (%s)\n", buf);

                } else {

                    ret = 0;

                }

            }

            break;

        }



        pkt= add_to_pktbuf(&ic->packet_buffer, &pkt1, &ic->packet_buffer_end);

        if(av_dup_packet(pkt) < 0) {

            return AVERROR(ENOMEM);

        }



        read_size += pkt->size;



        st = ic->streams[pkt->stream_index];

        if(st->codec_info_nb_frames>1) {

            if (st->time_base.den > 0 && av_rescale_q(info[st->index].codec_info_duration, st->time_base, AV_TIME_BASE_Q) >= ic->max_analyze_duration){

                av_log(ic, AV_LOG_WARNING, "max_analyze_duration reached\n");

                break;

            }

            info[st->index].codec_info_duration += pkt->duration;

        }

        {

            int index= pkt->stream_index;

            int64_t last= info[index].last_dts;

            int64_t duration= pkt->dts - last;



            if(pkt->dts != AV_NOPTS_VALUE && last != AV_NOPTS_VALUE && duration>0){

                double dur= duration * av_q2d(st->time_base);



//                if(st->codec->codec_type == AVMEDIA_TYPE_VIDEO)

//                    av_log(NULL, AV_LOG_ERROR, "%f\n", dur);

                if(info[index].duration_count < 2)

                    memset(info[index].duration_error, 0, sizeof(info[index].duration_error));

                for(i=1; i<MAX_STD_TIMEBASES; i++){

                    int framerate= get_std_framerate(i);

                    int ticks= lrintf(dur*framerate/(1001*12));

                    double error= dur - ticks*1001*12/(double)framerate;

                    info[index].duration_error[i] += error*error;

                }

                info[index].duration_count++;

                // ignore the first 4 values, they might have some random jitter

                if (info[index].duration_count > 3)

                    info[index].duration_gcd = av_gcd(info[index].duration_gcd, duration);

            }

            if(last == AV_NOPTS_VALUE || info[index].duration_count <= 1)

                info[pkt->stream_index].last_dts = pkt->dts;

        }

        if(st->parser && st->parser->parser->split && !st->codec->extradata){

            int i= st->parser->parser->split(st->codec, pkt->data, pkt->size);

            if(i){

                st->codec->extradata_size= i;

                st->codec->extradata= av_malloc(st->codec->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE);

                memcpy(st->codec->extradata, pkt->data, st->codec->extradata_size);

                memset(st->codec->extradata + i, 0, FF_INPUT_BUFFER_PADDING_SIZE);

            }

        }



        /* if still no information, we try to open the codec and to

           decompress the frame. We try to avoid that in most cases as

           it takes longer and uses more memory. For MPEG-4, we need to

           decompress for QuickTime. */

        if (!has_codec_parameters(st->codec) || !has_decode_delay_been_guessed(st))

            try_decode_frame(st, pkt);



        st->codec_info_nb_frames++;

        count++;

    }



    // close codecs which were opened in try_decode_frame()

    for(i=0;i<ic->nb_streams;i++) {

        st = ic->streams[i];

        if(st->codec->codec)

            avcodec_close(st->codec);

    }

    for(i=0;i<ic->nb_streams;i++) {

        st = ic->streams[i];

        if(st->codec_info_nb_frames>2 && !st->avg_frame_rate.num && info[i].codec_info_duration)

            av_reduce(&st->avg_frame_rate.num, &st->avg_frame_rate.den,

                     (st->codec_info_nb_frames-2)*(int64_t)st->time_base.den,

                      info[i].codec_info_duration*(int64_t)st->time_base.num, 60000);

        if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

            if(st->codec->codec_id == CODEC_ID_RAWVIDEO && !st->codec->codec_tag && !st->codec->bits_per_coded_sample)

                st->codec->codec_tag= avcodec_pix_fmt_to_codec_tag(st->codec->pix_fmt);



            // the check for tb_unreliable() is not completely correct, since this is not about handling

            // a unreliable/inexact time base, but a time base that is finer than necessary, as e.g.

            // ipmovie.c produces.

            if (tb_unreliable(st->codec) && info[i].duration_count > 15 && info[i].duration_gcd > 1 && !st->r_frame_rate.num)

                av_reduce(&st->r_frame_rate.num, &st->r_frame_rate.den, st->time_base.den, st->time_base.num * info[i].duration_gcd, INT_MAX);

            if(info[i].duration_count && !st->r_frame_rate.num

               && tb_unreliable(st->codec) /*&&

               //FIXME we should not special-case MPEG-2, but this needs testing with non-MPEG-2 ...

               st->time_base.num*duration_sum[i]/info[i].duration_count*101LL > st->time_base.den*/){

                int num = 0;

                double best_error= 2*av_q2d(st->time_base);

                best_error= best_error*best_error*info[i].duration_count*1000*12*30;



                for(j=1; j<MAX_STD_TIMEBASES; j++){

                    double error= info[i].duration_error[j] * get_std_framerate(j);

//                    if(st->codec->codec_type == AVMEDIA_TYPE_VIDEO)

//                        av_log(NULL, AV_LOG_ERROR, "%f %f\n", get_std_framerate(j) / 12.0/1001, error);

                    if(error < best_error){

                        best_error= error;

                        num = get_std_framerate(j);

                    }

                }

                // do not increase frame rate by more than 1 % in order to match a standard rate.

                if (num && (!st->r_frame_rate.num || (double)num/(12*1001) < 1.01 * av_q2d(st->r_frame_rate)))

                    av_reduce(&st->r_frame_rate.num, &st->r_frame_rate.den, num, 12*1001, INT_MAX);

            }



            if (!st->r_frame_rate.num){

                if(    st->codec->time_base.den * (int64_t)st->time_base.num

                    <= st->codec->time_base.num * st->codec->ticks_per_frame * (int64_t)st->time_base.den){

                    st->r_frame_rate.num = st->codec->time_base.den;

                    st->r_frame_rate.den = st->codec->time_base.num * st->codec->ticks_per_frame;

                }else{

                    st->r_frame_rate.num = st->time_base.den;

                    st->r_frame_rate.den = st->time_base.num;

                }

            }

        }else if(st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

            if(!st->codec->bits_per_coded_sample)

                st->codec->bits_per_coded_sample= av_get_bits_per_sample(st->codec->codec_id);

        }

    }



    av_estimate_timings(ic, old_offset);



    compute_chapters_end(ic);



#if 0

    /* correct DTS for B-frame streams with no timestamps */

    for(i=0;i<ic->nb_streams;i++) {

        st = ic->streams[i];

        if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

            if(b-frames){

                ppktl = &ic->packet_buffer;

                while(ppkt1){

                    if(ppkt1->stream_index != i)

                        continue;

                    if(ppkt1->pkt->dts < 0)

                        break;

                    if(ppkt1->pkt->pts != AV_NOPTS_VALUE)

                        break;

                    ppkt1->pkt->dts -= delta;

                    ppkt1= ppkt1->next;

                }

                if(ppkt1)

                    continue;

                st->cur_dts -= delta;

            }

        }

    }

#endif



    return ret;

}

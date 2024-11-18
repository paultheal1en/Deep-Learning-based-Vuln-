static void write_audio_frame(AVFormatContext *oc, AVStream *st)

{

    AVCodecContext *c;

    AVPacket pkt = { 0 }; // data and size must be 0;

    int got_packet, ret, dst_nb_samples;



    av_init_packet(&pkt);

    c = st->codec;



    get_audio_frame((int16_t *)src_samples_data[0], src_nb_samples, c->channels);



    /* convert samples from native format to destination codec format, using the resampler */

    if (swr_ctx) {

        /* compute destination number of samples */

        dst_nb_samples = av_rescale_rnd(swr_get_delay(swr_ctx, c->sample_rate) + src_nb_samples,

                                        c->sample_rate, c->sample_rate, AV_ROUND_UP);

        if (dst_nb_samples > max_dst_nb_samples) {

            av_free(dst_samples_data[0]);

            ret = av_samples_alloc(dst_samples_data, &dst_samples_linesize, c->channels,

                                   dst_nb_samples, c->sample_fmt, 0);

            if (ret < 0)

                exit(1);

            max_dst_nb_samples = dst_nb_samples;

            dst_samples_size = av_samples_get_buffer_size(NULL, c->channels, dst_nb_samples,

                                                          c->sample_fmt, 0);

        }



        /* convert to destination format */

        ret = swr_convert(swr_ctx,

                          dst_samples_data, dst_nb_samples,

                          (const uint8_t **)src_samples_data, src_nb_samples);

        if (ret < 0) {

            fprintf(stderr, "Error while converting\n");

            exit(1);

        }

    } else {

        dst_nb_samples = src_nb_samples;

    }



    audio_frame->nb_samples = dst_nb_samples;

    audio_frame->pts = av_rescale_q(samples_count, (AVRational){1, c->sample_rate}, c->time_base);

    avcodec_fill_audio_frame(audio_frame, c->channels, c->sample_fmt,

                             dst_samples_data[0], dst_samples_size, 0);

    samples_count += dst_nb_samples;



    ret = avcodec_encode_audio2(c, &pkt, audio_frame, &got_packet);

    if (ret < 0) {

        fprintf(stderr, "Error encoding audio frame: %s\n", av_err2str(ret));

        exit(1);

    }



    if (!got_packet)

        return;



    ret = write_frame(oc, &c->time_base, st, &pkt);

    if (ret != 0) {

        fprintf(stderr, "Error while writing audio frame: %s\n",

                av_err2str(ret));

        exit(1);

    }

}

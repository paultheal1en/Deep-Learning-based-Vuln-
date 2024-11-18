static int amovie_get_samples(AVFilterLink *outlink)

{

    MovieContext *movie = outlink->src->priv;

    AVPacket pkt;

    int ret, got_frame = 0;



    if (!movie->pkt.size && movie->is_done == 1)

        return AVERROR_EOF;



    /* check for another frame, in case the previous one was completely consumed */

    if (!movie->pkt.size) {

        while ((ret = av_read_frame(movie->format_ctx, &pkt)) >= 0) {

            // Is this a packet from the selected stream?

            if (pkt.stream_index != movie->stream_index) {

                av_free_packet(&pkt);

                continue;

            } else {

                movie->pkt0 = movie->pkt = pkt;

                break;

            }

        }



        if (ret == AVERROR_EOF) {

            movie->is_done = 1;

            return ret;

        }

    }



    /* decode and update the movie pkt */

    avcodec_get_frame_defaults(movie->frame);

    ret = avcodec_decode_audio4(movie->codec_ctx, movie->frame, &got_frame, &movie->pkt);

    if (ret < 0) {

        movie->pkt.size = 0;

        return ret;

    }

    movie->pkt.data += ret;

    movie->pkt.size -= ret;



    /* wrap the decoded data in a samplesref */

    if (got_frame) {

        int nb_samples = movie->frame->nb_samples;

        int data_size =

            av_samples_get_buffer_size(NULL, movie->codec_ctx->channels,

                                       nb_samples, movie->codec_ctx->sample_fmt, 1);

        if (data_size < 0)

            return data_size;

        movie->samplesref =

            ff_get_audio_buffer(outlink, AV_PERM_WRITE, nb_samples);

        memcpy(movie->samplesref->data[0], movie->frame->data[0], data_size);

        movie->samplesref->pts = movie->pkt.pts;

        movie->samplesref->pos = movie->pkt.pos;

        movie->samplesref->audio->sample_rate = movie->codec_ctx->sample_rate;

    }



    // We got it. Free the packet since we are returning

    if (movie->pkt.size <= 0)

        av_free_packet(&movie->pkt0);



    return 0;

}

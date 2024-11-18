static int film_read_packet(AVFormatContext *s,

                            AVPacket *pkt)

{

    FilmDemuxContext *film = s->priv_data;

    AVIOContext *pb = s->pb;

    film_sample *sample;

    int ret = 0;

    int i;

    int left, right;



    if (film->current_sample >= film->sample_count)




    sample = &film->sample_table[film->current_sample];



    /* position the stream (will probably be there anyway) */

    avio_seek(pb, sample->sample_offset, SEEK_SET);



    /* do a special song and dance when loading FILM Cinepak chunks */

    if ((sample->stream == film->video_stream_index) &&

        (film->video_type == CODEC_ID_CINEPAK)) {

        pkt->pos= avio_tell(pb);

        if (av_new_packet(pkt, sample->sample_size))

            return AVERROR(ENOMEM);

        avio_read(pb, pkt->data, sample->sample_size);

    } else if ((sample->stream == film->audio_stream_index) &&

        (film->audio_channels == 2) &&

        (film->audio_type != CODEC_ID_ADPCM_ADX)) {

        /* stereo PCM needs to be interleaved */





        if (av_new_packet(pkt, sample->sample_size))

            return AVERROR(ENOMEM);



        /* make sure the interleave buffer is large enough */

        if (sample->sample_size > film->stereo_buffer_size) {

            av_free(film->stereo_buffer);

            film->stereo_buffer_size = sample->sample_size;

            film->stereo_buffer = av_malloc(film->stereo_buffer_size);

            if (!film->stereo_buffer) {

                film->stereo_buffer_size = 0;

                return AVERROR(ENOMEM);

            }

        }



        pkt->pos= avio_tell(pb);

        ret = avio_read(pb, film->stereo_buffer, sample->sample_size);

        if (ret != sample->sample_size)

            ret = AVERROR(EIO);



        left = 0;

        right = sample->sample_size / 2;

        for (i = 0; i < sample->sample_size; ) {

            if (film->audio_bits == 8) {

                pkt->data[i++] = film->stereo_buffer[left++];

                pkt->data[i++] = film->stereo_buffer[right++];

            } else {

                pkt->data[i++] = film->stereo_buffer[left++];

                pkt->data[i++] = film->stereo_buffer[left++];

                pkt->data[i++] = film->stereo_buffer[right++];

                pkt->data[i++] = film->stereo_buffer[right++];

            }

        }

    } else {

        ret= av_get_packet(pb, pkt, sample->sample_size);

        if (ret != sample->sample_size)

            ret = AVERROR(EIO);

    }



    pkt->stream_index = sample->stream;

    pkt->pts = sample->pts;



    film->current_sample++;



    return ret;

}
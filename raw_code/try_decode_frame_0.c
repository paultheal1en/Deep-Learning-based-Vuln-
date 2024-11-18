static int try_decode_frame(AVStream *st, AVPacket *avpkt, AVDictionary **options)

{

    AVCodec *codec;

    int got_picture = 1, ret = 0;

    AVFrame picture;

    AVPacket pkt = *avpkt;



    if(!st->codec->codec){

        AVDictionary *thread_opt = NULL;



        codec = avcodec_find_decoder(st->codec->codec_id);

        if (!codec)

            return -1;



        /* force thread count to 1 since the h264 decoder will not extract SPS

         *  and PPS to extradata during multi-threaded decoding */

        av_dict_set(options ? options : &thread_opt, "threads", "1", 0);

        ret = avcodec_open2(st->codec, codec, options ? options : &thread_opt);

        if (!options)

            av_dict_free(&thread_opt);

        if (ret < 0)

            return ret;

    }



    while ((pkt.size > 0 || (!pkt.data && got_picture)) &&

           ret >= 0 &&

           (!has_codec_parameters(st->codec)  ||

           !has_decode_delay_been_guessed(st) ||

           (!st->codec_info_nb_frames && st->codec->codec->capabilities & CODEC_CAP_CHANNEL_CONF))) {

        got_picture = 0;

        avcodec_get_frame_defaults(&picture);

        switch(st->codec->codec_type) {

        case AVMEDIA_TYPE_VIDEO:

            ret = avcodec_decode_video2(st->codec, &picture,

                                        &got_picture, &pkt);

            break;

        case AVMEDIA_TYPE_AUDIO:

            ret = avcodec_decode_audio4(st->codec, &picture, &got_picture, &pkt);

            break;

        default:

            break;

        }

        if (ret >= 0) {

            if (got_picture)

                st->info->nb_decoded_frames++;

            pkt.data += ret;

            pkt.size -= ret;

            ret       = got_picture;

        }

    }

    return ret;

}

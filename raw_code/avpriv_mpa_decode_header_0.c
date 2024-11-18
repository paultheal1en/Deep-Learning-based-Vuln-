int avpriv_mpa_decode_header(AVCodecContext *avctx, uint32_t head, int *sample_rate, int *channels, int *frame_size, int *bit_rate)

{

    MPADecodeHeader s1, *s = &s1;



    if (ff_mpa_check_header(head) != 0)

        return -1;



    if (avpriv_mpegaudio_decode_header(s, head) != 0) {

        return -1;

    }



    switch(s->layer) {

    case 1:

        avctx->codec_id = AV_CODEC_ID_MP1;

        *frame_size = 384;

        break;

    case 2:

        avctx->codec_id = AV_CODEC_ID_MP2;

        *frame_size = 1152;

        break;

    default:

    case 3:

        if (avctx->codec_id != AV_CODEC_ID_MP3ADU)

            avctx->codec_id = AV_CODEC_ID_MP3;

        if (s->lsf)

            *frame_size = 576;

        else

            *frame_size = 1152;

        break;

    }



    *sample_rate = s->sample_rate;

    *channels = s->nb_channels;

    *bit_rate = s->bit_rate;

    return s->frame_size;

}

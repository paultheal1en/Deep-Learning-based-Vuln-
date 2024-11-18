static int encode_init(AVCodecContext * avctx){

    WMACodecContext *s = avctx->priv_data;

    int i, flags1, flags2;

    uint8_t *extradata;



    s->avctx = avctx;



    if(avctx->channels > MAX_CHANNELS) {

        av_log(avctx, AV_LOG_ERROR, "too many channels: got %i, need %i or fewer",

               avctx->channels, MAX_CHANNELS);

        return AVERROR(EINVAL);

    }



    if (avctx->sample_rate > 48000) {

        av_log(avctx, AV_LOG_ERROR, "sample rate is too high: %d > 48kHz",

               avctx->sample_rate);

        return AVERROR(EINVAL);

    }



    if(avctx->bit_rate < 24*1000) {

        av_log(avctx, AV_LOG_ERROR, "bitrate too low: got %i, need 24000 or higher\n",

               avctx->bit_rate);

        return AVERROR(EINVAL);

    }



    /* extract flag infos */

    flags1 = 0;

    flags2 = 1;

    if (avctx->codec->id == CODEC_ID_WMAV1) {

        extradata= av_malloc(4);

        avctx->extradata_size= 4;

        AV_WL16(extradata, flags1);

        AV_WL16(extradata+2, flags2);

    } else if (avctx->codec->id == CODEC_ID_WMAV2) {

        extradata= av_mallocz(10);

        avctx->extradata_size= 10;

        AV_WL32(extradata, flags1);

        AV_WL16(extradata+4, flags2);

    }else

        av_assert0(0);

    avctx->extradata= extradata;

    s->use_exp_vlc = flags2 & 0x0001;

    s->use_bit_reservoir = flags2 & 0x0002;

    s->use_variable_block_len = flags2 & 0x0004;

    if (avctx->channels == 2)

        s->ms_stereo = 1;



    ff_wma_init(avctx, flags2);



    /* init MDCT */

    for(i = 0; i < s->nb_block_sizes; i++)

        ff_mdct_init(&s->mdct_ctx[i], s->frame_len_bits - i + 1, 0, 1.0);



    s->block_align     = avctx->bit_rate * (int64_t)s->frame_len /

                         (avctx->sample_rate * 8);

    s->block_align     = FFMIN(s->block_align, MAX_CODED_SUPERFRAME_SIZE);

    avctx->block_align = s->block_align;

    avctx->bit_rate    = avctx->block_align * 8LL * avctx->sample_rate /

                         s->frame_len;

//av_log(NULL, AV_LOG_ERROR, "%d %d %d %d\n", s->block_align, avctx->bit_rate, s->frame_len, avctx->sample_rate);

    avctx->frame_size = avctx->delay = s->frame_len;



#if FF_API_OLD_ENCODE_AUDIO

    avctx->coded_frame = &s->frame;

    avcodec_get_frame_defaults(avctx->coded_frame);

#endif



    return 0;

}

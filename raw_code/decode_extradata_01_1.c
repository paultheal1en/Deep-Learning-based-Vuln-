static int decode_extradata(AVFormatContext *s, ADTSContext *adts, uint8_t *buf, int size)

{

    GetBitContext gb;



    init_get_bits(&gb, buf, size * 8);

    adts->objecttype = get_bits(&gb, 5) - 1;

    adts->sample_rate_index = get_bits(&gb, 4);

    adts->channel_conf = get_bits(&gb, 4);



    if (adts->objecttype > 3) {

        av_log(s, AV_LOG_ERROR, "MPEG-4 AOT %d is not allowed in ADTS\n", adts->objecttype);

        return -1;

    }

    if (adts->sample_rate_index == 15) {

        av_log(s, AV_LOG_ERROR, "Escape sample rate index illegal in ADTS\n");

        return -1;

    }

    if (adts->channel_conf == 0) {

        ff_log_missing_feature(s, "PCE based channel configuration", 0);

        return -1;

    }



    adts->write_adts = 1;



    return 0;

}

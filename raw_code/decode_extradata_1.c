static int decode_extradata(ADTSContext *adts, uint8_t *buf, int size)

{

    GetBitContext gb;



    init_get_bits(&gb, buf, size * 8);

    adts->objecttype = get_bits(&gb, 5) - 1;

    adts->sample_rate_index = get_bits(&gb, 4);

    adts->channel_conf = get_bits(&gb, 4);



    adts->write_adts = 1;



    return 0;

}

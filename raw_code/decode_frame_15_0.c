static int decode_frame(FLACContext *s, int alloc_data_size)

{

    int bs_code, sr_code, bps_code, i;

    int ch_mode, bps, blocksize, samplerate;

    GetBitContext *gb = &s->gb;



    /* frame sync code */

    skip_bits(&s->gb, 16);



    /* block size and sample rate codes */

    bs_code = get_bits(gb, 4);

    sr_code = get_bits(gb, 4);



    /* channels and decorrelation */

    ch_mode = get_bits(gb, 4);

    if (ch_mode < FLAC_MAX_CHANNELS && s->channels == ch_mode+1) {

        ch_mode = FLAC_CHMODE_INDEPENDENT;

    } else if (ch_mode > FLAC_CHMODE_MID_SIDE || s->channels != 2) {

        av_log(s->avctx, AV_LOG_ERROR, "unsupported channel assignment %d (channels=%d)\n",

               ch_mode, s->channels);

        return -1;

    }



    /* bits per sample */

    bps_code = get_bits(gb, 3);

    if (bps_code == 0)

        bps= s->bps;

    else if ((bps_code != 3) && (bps_code != 7))

        bps = sample_size_table[bps_code];

    else {

        av_log(s->avctx, AV_LOG_ERROR, "invalid sample size code (%d)\n",

               bps_code);

        return -1;

    }

    if (bps > 16) {

        s->avctx->sample_fmt = SAMPLE_FMT_S32;

        s->sample_shift = 32 - bps;

        s->is32 = 1;

    } else {

        s->avctx->sample_fmt = SAMPLE_FMT_S16;

        s->sample_shift = 16 - bps;

        s->is32 = 0;

    }

    s->bps = s->avctx->bits_per_raw_sample = bps;



    /* reserved bit */

    if (get_bits1(gb)) {

        av_log(s->avctx, AV_LOG_ERROR, "broken stream, invalid padding\n");

        return -1;

    }



    /* sample or frame count */

    if (get_utf8(gb) < 0) {

        av_log(s->avctx, AV_LOG_ERROR, "utf8 fscked\n");

        return -1;

    }



    /* blocksize */

    if (bs_code == 0) {

        av_log(s->avctx, AV_LOG_ERROR, "reserved blocksize code: 0\n");

        return -1;

    } else if (bs_code == 6)

        blocksize = get_bits(gb, 8)+1;

    else if (bs_code == 7)

        blocksize = get_bits(gb, 16)+1;

    else

        blocksize = ff_flac_blocksize_table[bs_code];



    if (blocksize > s->max_blocksize) {

        av_log(s->avctx, AV_LOG_ERROR, "blocksize %d > %d\n", blocksize,

               s->max_blocksize);

        return -1;

    }



    if (blocksize * s->channels * (s->is32 ? 4 : 2) > alloc_data_size)

        return -1;



    /* sample rate */

    if (sr_code == 0)

        samplerate= s->samplerate;

    else if (sr_code < 12)

        samplerate = ff_flac_sample_rate_table[sr_code];

    else if (sr_code == 12)

        samplerate = get_bits(gb, 8) * 1000;

    else if (sr_code == 13)

        samplerate = get_bits(gb, 16);

    else if (sr_code == 14)

        samplerate = get_bits(gb, 16) * 10;

    else {

        av_log(s->avctx, AV_LOG_ERROR, "illegal sample rate code %d\n",

               sr_code);

        return -1;

    }



    /* header CRC-8 check */

    skip_bits(gb, 8);

    if (av_crc(av_crc_get_table(AV_CRC_8_ATM), 0, gb->buffer,

               get_bits_count(gb)/8)) {

        av_log(s->avctx, AV_LOG_ERROR, "header crc mismatch\n");

        return -1;

    }



    s->blocksize    = blocksize;

    s->samplerate   = samplerate;

    s->bps          = bps;

    s->ch_mode      = ch_mode;



//    dump_headers(s->avctx, (FLACStreaminfo *)s);



    /* subframes */

    for (i = 0; i < s->channels; i++) {

        if (decode_subframe(s, i) < 0)

            return -1;

    }



    align_get_bits(gb);



    /* frame footer */

    skip_bits(gb, 16); /* data crc */



    return 0;

}

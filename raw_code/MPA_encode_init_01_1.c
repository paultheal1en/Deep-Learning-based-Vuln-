static av_cold int MPA_encode_init(AVCodecContext *avctx)

{

    MpegAudioContext *s = avctx->priv_data;

    int freq = avctx->sample_rate;

    int bitrate = avctx->bit_rate;

    int channels = avctx->channels;

    int i, v, table;

    float a;



    if (channels <= 0 || channels > 2){

        av_log(avctx, AV_LOG_ERROR, "encoding %d channel(s) is not allowed in mp2\n", channels);

        return AVERROR(EINVAL);

    }

    bitrate = bitrate / 1000;

    s->nb_channels = channels;

    avctx->frame_size = MPA_FRAME_SIZE;

    avctx->delay      = 512 - 32 + 1;



    /* encoding freq */

    s->lsf = 0;

    for(i=0;i<3;i++) {

        if (avpriv_mpa_freq_tab[i] == freq)

            break;

        if ((avpriv_mpa_freq_tab[i] / 2) == freq) {

            s->lsf = 1;

            break;

        }

    }

    if (i == 3){

        av_log(avctx, AV_LOG_ERROR, "Sampling rate %d is not allowed in mp2\n", freq);

        return AVERROR(EINVAL);

    }

    s->freq_index = i;



    /* encoding bitrate & frequency */

    for(i=0;i<15;i++) {

        if (avpriv_mpa_bitrate_tab[s->lsf][1][i] == bitrate)

            break;

    }

    if (i == 15){

        av_log(avctx, AV_LOG_ERROR, "bitrate %d is not allowed in mp2\n", bitrate);

        return AVERROR(EINVAL);

    }

    s->bitrate_index = i;



    /* compute total header size & pad bit */



    a = (float)(bitrate * 1000 * MPA_FRAME_SIZE) / (freq * 8.0);

    s->frame_size = ((int)a) * 8;



    /* frame fractional size to compute padding */

    s->frame_frac = 0;

    s->frame_frac_incr = (int)((a - floor(a)) * 65536.0);



    /* select the right allocation table */

    table = ff_mpa_l2_select_table(bitrate, s->nb_channels, freq, s->lsf);



    /* number of used subbands */

    s->sblimit = ff_mpa_sblimit_table[table];

    s->alloc_table = ff_mpa_alloc_tables[table];



    av_dlog(avctx, "%d kb/s, %d Hz, frame_size=%d bits, table=%d, padincr=%x\n",

            bitrate, freq, s->frame_size, table, s->frame_frac_incr);



    for(i=0;i<s->nb_channels;i++)

        s->samples_offset[i] = 0;



    for(i=0;i<257;i++) {

        int v;

        v = ff_mpa_enwindow[i];

#if WFRAC_BITS != 16

        v = (v + (1 << (16 - WFRAC_BITS - 1))) >> (16 - WFRAC_BITS);

#endif

        s->filter_bank[i] = v;

        if ((i & 63) != 0)

            v = -v;

        if (i != 0)

            s->filter_bank[512 - i] = v;

    }



    for(i=0;i<64;i++) {

        v = (int)(exp2((3 - i) / 3.0) * (1 << 20));

        if (v <= 0)

            v = 1;

        s->scale_factor_table[i] = v;

#if USE_FLOATS

        s->scale_factor_inv_table[i] = exp2(-(3 - i) / 3.0) / (float)(1 << 20);

#else

#define P 15

        s->scale_factor_shift[i] = 21 - P - (i / 3);

        s->scale_factor_mult[i] = (1 << P) * exp2((i % 3) / 3.0);

#endif

    }

    for(i=0;i<128;i++) {

        v = i - 64;

        if (v <= -3)

            v = 0;

        else if (v < 0)

            v = 1;

        else if (v == 0)

            v = 2;

        else if (v < 3)

            v = 3;

        else

            v = 4;

        s->scale_diff_table[i] = v;

    }



    for(i=0;i<17;i++) {

        v = ff_mpa_quant_bits[i];

        if (v < 0)

            v = -v;

        else

            v = v * 3;

        s->total_quant_bits[i] = 12 * v;

    }



    return 0;

}

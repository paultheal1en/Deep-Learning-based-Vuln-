static int decode_channel_residues(WmallDecodeCtx *s, int ch, int tile_size)

{

    int i = 0;

    unsigned int ave_mean;

    s->transient[ch] = get_bits1(&s->gb);

    if (s->transient[ch]) {

        s->transient_pos[ch] = get_bits(&s->gb, av_log2(tile_size));

        if (s->transient_pos[ch])

            s->transient[ch] = 0;

        s->channel[ch].transient_counter =

            FFMAX(s->channel[ch].transient_counter, s->samples_per_frame / 2);

    } else if (s->channel[ch].transient_counter)

        s->transient[ch] = 1;



    if (s->seekable_tile) {

        ave_mean = get_bits(&s->gb, s->bits_per_sample);

        s->ave_sum[ch] = ave_mean << (s->movave_scaling + 1);

    }



    if (s->seekable_tile) {

        if (s->do_inter_ch_decorr)

            s->channel_residues[ch][0] = get_sbits(&s->gb, s->bits_per_sample + 1);

        else

            s->channel_residues[ch][0] = get_sbits(&s->gb, s->bits_per_sample);

        i++;

    }

    for (; i < tile_size; i++) {

        int quo = 0, rem, rem_bits, residue;

        while(get_bits1(&s->gb)) {

            quo++;

            if (get_bits_left(&s->gb) <= 0)

                return -1;

        }

        if (quo >= 32)

            quo += get_bits_long(&s->gb, get_bits(&s->gb, 5) + 1);



        ave_mean = (s->ave_sum[ch] + (1 << s->movave_scaling)) >> (s->movave_scaling + 1);

        if (ave_mean <= 1)

            residue = quo;

        else {

            rem_bits = av_ceil_log2(ave_mean);

            rem      = rem_bits ? get_bits(&s->gb, rem_bits) : 0;

            residue  = (quo << rem_bits) + rem;

        }



        s->ave_sum[ch] = residue + s->ave_sum[ch] -

                         (s->ave_sum[ch] >> s->movave_scaling);



        if (residue & 1)

            residue = -(residue >> 1) - 1;

        else

            residue = residue >> 1;

        s->channel_residues[ch][i] = residue;

    }



    return 0;



}

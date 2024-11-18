static void aw_pulse_set2(WMAVoiceContext *s, GetBitContext *gb,

                          int block_idx, AMRFixed *fcb)

{

    uint16_t use_mask_mem[9]; // only 5 are used, rest is padding

    uint16_t *use_mask = use_mask_mem + 2;

    /* in this function, idx is the index in the 80-bit (+ padding) use_mask

     * bit-array. Since use_mask consists of 16-bit values, the lower 4 bits

     * of idx are the position of the bit within a particular item in the

     * array (0 being the most significant bit, and 15 being the least

     * significant bit), and the remainder (>> 4) is the index in the

     * use_mask[]-array. This is faster and uses less memory than using a

     * 80-byte/80-int array. */

    int pulse_off = s->aw_first_pulse_off[block_idx],

        pulse_start, n, idx, range, aidx, start_off = 0;



    /* set offset of first pulse to within this block */

    if (s->aw_n_pulses[block_idx] > 0)

        while (pulse_off + s->aw_pulse_range < 1)

            pulse_off += fcb->pitch_lag;



    /* find range per pulse */

    if (s->aw_n_pulses[0] > 0) {

        if (block_idx == 0) {

            range = 32;

        } else /* block_idx = 1 */ {

            range = 8;

            if (s->aw_n_pulses[block_idx] > 0)

                pulse_off = s->aw_next_pulse_off_cache;

        }

    } else

        range = 16;

    pulse_start = s->aw_n_pulses[block_idx] > 0 ? pulse_off - range / 2 : 0;



    /* aw_pulse_set1() already applies pulses around pulse_off (to be exactly,

     * in the range of [pulse_off, pulse_off + s->aw_pulse_range], and thus

     * we exclude that range from being pulsed again in this function. */

    memset(&use_mask[-2], 0, 2 * sizeof(use_mask[0]));

    memset( use_mask,   -1, 5 * sizeof(use_mask[0]));

    memset(&use_mask[5], 0, 2 * sizeof(use_mask[0]));

    if (s->aw_n_pulses[block_idx] > 0)

        for (idx = pulse_off; idx < MAX_FRAMESIZE / 2; idx += fcb->pitch_lag) {

            int excl_range         = s->aw_pulse_range; // always 16 or 24

            uint16_t *use_mask_ptr = &use_mask[idx >> 4];

            int first_sh           = 16 - (idx & 15);

            *use_mask_ptr++       &= 0xFFFF << first_sh;

            excl_range            -= first_sh;

            if (excl_range >= 16) {

                *use_mask_ptr++    = 0;

                *use_mask_ptr     &= 0xFFFF >> (excl_range - 16);

            } else

                *use_mask_ptr     &= 0xFFFF >> excl_range;

        }



    /* find the 'aidx'th offset that is not excluded */

    aidx = get_bits(gb, s->aw_n_pulses[0] > 0 ? 5 - 2 * block_idx : 4);

    for (n = 0; n <= aidx; pulse_start++) {

        for (idx = pulse_start; idx < 0; idx += fcb->pitch_lag) ;

        if (idx >= MAX_FRAMESIZE / 2) { // find from zero

            if (use_mask[0])      idx = 0x0F;

            else if (use_mask[1]) idx = 0x1F;

            else if (use_mask[2]) idx = 0x2F;

            else if (use_mask[3]) idx = 0x3F;

            else if (use_mask[4]) idx = 0x4F;

            else                  return;

            idx -= av_log2_16bit(use_mask[idx >> 4]);

        }

        if (use_mask[idx >> 4] & (0x8000 >> (idx & 15))) {

            use_mask[idx >> 4] &= ~(0x8000 >> (idx & 15));

            n++;

            start_off = idx;

        }

    }



    fcb->x[fcb->n] = start_off;

    fcb->y[fcb->n] = get_bits1(gb) ? -1.0 : 1.0;

    fcb->n++;



    /* set offset for next block, relative to start of that block */

    n = (MAX_FRAMESIZE / 2 - start_off) % fcb->pitch_lag;

    s->aw_next_pulse_off_cache = n ? fcb->pitch_lag - n : 0;

}

static int decode_pulses(Pulse *pulse, GetBitContext *gb,

                         const uint16_t *swb_offset, int num_swb)

{

    int i, pulse_swb;

    pulse->num_pulse = get_bits(gb, 2) + 1;

    pulse_swb        = get_bits(gb, 6);

    if (pulse_swb >= num_swb)

        return -1;

    pulse->pos[0]    = swb_offset[pulse_swb];

    pulse->pos[0]   += get_bits(gb, 5);

    if (pulse->pos[0] > 1023)

        return -1;

    pulse->amp[0]    = get_bits(gb, 4);

    for (i = 1; i < pulse->num_pulse; i++) {

        pulse->pos[i] = get_bits(gb, 5) + pulse->pos[i - 1];

        if (pulse->pos[i] > 1023)

            return -1;

        pulse->amp[i] = get_bits(gb, 4);

    }

    return 0;

}

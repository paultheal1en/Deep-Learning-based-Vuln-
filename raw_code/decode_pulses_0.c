static void decode_pulses(Pulse * pulse, GetBitContext * gb, const uint16_t * swb_offset) {

    int i;

    pulse->num_pulse = get_bits(gb, 2) + 1;

    pulse->pos[0]    = swb_offset[get_bits(gb, 6)];

    pulse->pos[0]   += get_bits(gb, 5);

    pulse->amp[0]    = get_bits(gb, 4);

    for (i = 1; i < pulse->num_pulse; i++) {

        pulse->pos[i] = get_bits(gb, 5) + pulse->pos[i-1];

        pulse->amp[i] = get_bits(gb, 4);

    }

}

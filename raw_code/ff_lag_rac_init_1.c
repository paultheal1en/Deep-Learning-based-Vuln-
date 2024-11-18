void ff_lag_rac_init(lag_rac *l, GetBitContext *gb, int length)

{

    int i, j;



    /* According to reference decoder "1st byte is garbage",

     * however, it gets skipped by the call to align_get_bits()

     */

    align_get_bits(gb);

    l->bytestream_start =

    l->bytestream       = gb->buffer + get_bits_count(gb) / 8;

    l->bytestream_end   = l->bytestream_start + length;



    l->range        = 0x80;

    l->low          = *l->bytestream >> 1;

    l->hash_shift   = FFMAX(l->scale - 8, 0);



    for (i = j = 0; i < 256; i++) {

        unsigned r = i << l->hash_shift;

        while (l->prob[j + 1] <= r)

            j++;

        l->range_hash[i] = j;

    }



    /* Add conversion factor to hash_shift so we don't have to in lag_get_rac. */

    l->hash_shift += 23;

}

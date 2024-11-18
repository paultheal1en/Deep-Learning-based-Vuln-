void ff_ac3_bit_alloc_calc_bap(int16_t *mask, int16_t *psd, int start, int end,

                               int snr_offset, int floor,

                               const uint8_t *bap_tab, uint8_t *bap)

{

    int i, j, end1, v, address;



    /* special case, if snr offset is -960, set all bap's to zero */

    if (snr_offset == -960) {

        memset(bap, 0, 256);

        return;

    }



    i = start;

    j = bin_to_band_tab[start];

    do {

        v = (FFMAX(mask[j] - snr_offset - floor, 0) & 0x1FE0) + floor;

        end1 = FFMIN(band_start_tab[j] + ff_ac3_critical_band_size_tab[j], end);

        for (; i < end1; i++) {

            address = av_clip((psd[i] - v) >> 5, 0, 63);

            bap[i] = bap_tab[address];

        }

    } while (end > band_start_tab[j++]);

}

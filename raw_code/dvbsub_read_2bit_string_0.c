static int dvbsub_read_2bit_string(uint8_t *destbuf, int dbuf_len,

                                   const uint8_t **srcbuf, int buf_size,

                                   int non_mod, uint8_t *map_table)

{

    GetBitContext gb;



    int bits;

    int run_length;

    int pixels_read = 0;



    init_get_bits(&gb, *srcbuf, buf_size << 3);



    while (get_bits_count(&gb) < buf_size << 3 && pixels_read < dbuf_len) {

        bits = get_bits(&gb, 2);



        if (bits) {

            if (non_mod != 1 || bits != 1) {

                if (map_table)

                    *destbuf++ = map_table[bits];

                else

                    *destbuf++ = bits;

            }

            pixels_read++;

        } else {

            bits = get_bits1(&gb);

            if (bits == 1) {

                run_length = get_bits(&gb, 3) + 3;

                bits = get_bits(&gb, 2);



                if (non_mod == 1 && bits == 1)

                    pixels_read += run_length;

                else {

                    if (map_table)

                        bits = map_table[bits];

                    while (run_length-- > 0 && pixels_read < dbuf_len) {

                        *destbuf++ = bits;

                        pixels_read++;

                    }

                }

            } else {

                bits = get_bits1(&gb);

                if (bits == 0) {

                    bits = get_bits(&gb, 2);

                    if (bits == 2) {

                        run_length = get_bits(&gb, 4) + 12;

                        bits = get_bits(&gb, 2);



                        if (non_mod == 1 && bits == 1)

                            pixels_read += run_length;

                        else {

                            if (map_table)

                                bits = map_table[bits];

                            while (run_length-- > 0 && pixels_read < dbuf_len) {

                                *destbuf++ = bits;

                                pixels_read++;

                            }

                        }

                    } else if (bits == 3) {

                        run_length = get_bits(&gb, 8) + 29;

                        bits = get_bits(&gb, 2);



                        if (non_mod == 1 && bits == 1)

                            pixels_read += run_length;

                        else {

                            if (map_table)

                                bits = map_table[bits];

                            while (run_length-- > 0 && pixels_read < dbuf_len) {

                                *destbuf++ = bits;

                                pixels_read++;

                            }

                        }

                    } else if (bits == 1) {

                        pixels_read += 2;

                        if (map_table)

                            bits = map_table[0];

                        else

                            bits = 0;

                        if (pixels_read <= dbuf_len) {

                            *destbuf++ = bits;

                            *destbuf++ = bits;

                        }

                    } else {

                        (*srcbuf) += (get_bits_count(&gb) + 7) >> 3;

                        return pixels_read;

                    }

                } else {

                    if (map_table)

                        bits = map_table[0];

                    else

                        bits = 0;

                    *destbuf++ = bits;

                    pixels_read++;

                }

            }

        }

    }



    if (get_bits(&gb, 6))

        av_log(0, AV_LOG_ERROR, "DVBSub error: line overflow\n");



    (*srcbuf) += (get_bits_count(&gb) + 7) >> 3;



    return pixels_read;

}

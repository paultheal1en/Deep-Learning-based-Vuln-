static inline void dv_decode_video_segment(DVVideoDecodeContext *s, 

                                           UINT8 *buf_ptr1, 

                                           const UINT16 *mb_pos_ptr)

{

    int quant, dc, dct_mode, class1, j;

    int mb_index, mb_x, mb_y, v, last_index;

    DCTELEM *block, *block1;

    int c_offset, bits_left;

    UINT8 *y_ptr;

    BlockInfo mb_data[5 * 6], *mb, *mb1;

    void (*idct_put)(UINT8 *dest, int line_size, DCTELEM *block);

    UINT8 *buf_ptr;

    PutBitContext pb, vs_pb;

    UINT8 mb_bit_buffer[80 + 4]; /* allow some slack */

    int mb_bit_count;

    UINT8 vs_bit_buffer[5 * 80 + 4]; /* allow some slack */

    int vs_bit_count;

    

    memset(s->block, 0, sizeof(s->block));



    /* pass 1 : read DC and AC coefficients in blocks */

    buf_ptr = buf_ptr1;

    block1 = &s->block[0][0];

    mb1 = mb_data;

    init_put_bits(&vs_pb, vs_bit_buffer, 5 * 80, NULL, NULL);

    vs_bit_count = 0;

    for(mb_index = 0; mb_index < 5; mb_index++) {

        /* skip header */

        quant = buf_ptr[3] & 0x0f;

        buf_ptr += 4;

        init_put_bits(&pb, mb_bit_buffer, 80, NULL, NULL);

        mb_bit_count = 0;

        mb = mb1;

        block = block1;

        for(j = 0;j < 6; j++) {

            /* NOTE: size is not important here */

            init_get_bits(&s->gb, buf_ptr, 14);

            

            /* get the dc */

            dc = get_bits(&s->gb, 9);

            dc = (dc << (32 - 9)) >> (32 - 9);

            dct_mode = get_bits1(&s->gb);

            mb->dct_mode = dct_mode;

            mb->scan_table = s->dv_zigzag[dct_mode];

            class1 = get_bits(&s->gb, 2);

            mb->shift_offset = (class1 == 3);

            mb->shift_table = s->dv_shift[dct_mode]

                [quant + dv_quant_offset[class1]];

            dc = dc << 2;

            /* convert to unsigned because 128 is not added in the

               standard IDCT */

            dc += 1024;

            block[0] = dc;

            last_index = block_sizes[j];

            buf_ptr += last_index >> 3;

            mb->pos = 0;

            mb->partial_bit_count = 0;



            dv_decode_ac(s, mb, block, last_index);



            /* write the remaining bits  in a new buffer only if the

               block is finished */

            bits_left = last_index - s->gb.index;

            if (mb->eob_reached) {

                mb->partial_bit_count = 0;

                mb_bit_count += bits_left;

                bit_copy(&pb, &s->gb, bits_left);

            } else {

                /* should be < 16 bits otherwise a codeword could have

                   been parsed */

                mb->partial_bit_count = bits_left;

                mb->partial_bit_buffer = get_bits(&s->gb, bits_left);

            }

            block += 64;

            mb++;

        }

        

        flush_put_bits(&pb);



        /* pass 2 : we can do it just after */

#ifdef VLC_DEBUG

        printf("***pass 2 size=%d\n", mb_bit_count);

#endif

        block = block1;

        mb = mb1;

        init_get_bits(&s->gb, mb_bit_buffer, 80);

        for(j = 0;j < 6; j++) {

            if (!mb->eob_reached && s->gb.index < mb_bit_count) {

                dv_decode_ac(s, mb, block, mb_bit_count);

                /* if still not finished, no need to parse other blocks */

                if (!mb->eob_reached) {

                    /* we could not parse the current AC coefficient,

                       so we add the remaining bytes */

                    bits_left = mb_bit_count - s->gb.index;

                    if (bits_left > 0) {

                        mb->partial_bit_count += bits_left;

                        mb->partial_bit_buffer = 

                            (mb->partial_bit_buffer << bits_left) | 

                            get_bits(&s->gb, bits_left);

                    }

                    goto next_mb;

                }

            }

            block += 64;

            mb++;

        }

        /* all blocks are finished, so the extra bytes can be used at

           the video segment level */

        bits_left = mb_bit_count - s->gb.index;

        vs_bit_count += bits_left;

        bit_copy(&vs_pb, &s->gb, bits_left);

    next_mb:

        mb1 += 6;

        block1 += 6 * 64;

    }



    /* we need a pass other the whole video segment */

    flush_put_bits(&vs_pb);

        

#ifdef VLC_DEBUG

    printf("***pass 3 size=%d\n", vs_bit_count);

#endif

    block = &s->block[0][0];

    mb = mb_data;

    init_get_bits(&s->gb, vs_bit_buffer, 5 * 80);

    for(mb_index = 0; mb_index < 5; mb_index++) {

        for(j = 0;j < 6; j++) {

            if (!mb->eob_reached) {

#ifdef VLC_DEBUG

                printf("start %d:%d\n", mb_index, j);

#endif

                dv_decode_ac(s, mb, block, vs_bit_count);

            }

            block += 64;

            mb++;

        }

    }

    

    /* compute idct and place blocks */

    block = &s->block[0][0];

    mb = mb_data;

    for(mb_index = 0; mb_index < 5; mb_index++) {

        v = *mb_pos_ptr++;

        mb_x = v & 0xff;

        mb_y = v >> 8;

        y_ptr = s->current_picture[0] + (mb_y * s->linesize[0] * 8) + (mb_x * 8);

        if (s->sampling_411)

            c_offset = (mb_y * s->linesize[1] * 8) + ((mb_x >> 2) * 8);

        else

            c_offset = ((mb_y >> 1) * s->linesize[1] * 8) + ((mb_x >> 1) * 8);

        for(j = 0;j < 6; j++) {

            idct_put = s->idct_put[mb->dct_mode];

            if (j < 4) {

                if (s->sampling_411 && mb_x < (704 / 8)) {

                    /* NOTE: at end of line, the macroblock is handled as 420 */

                    idct_put(y_ptr + (j * 8), s->linesize[0], block);

                } else {

                    idct_put(y_ptr + ((j & 1) * 8) + ((j >> 1) * 8 * s->linesize[0]),

                             s->linesize[0], block);

                }

            } else {

                if (s->sampling_411 && mb_x >= (704 / 8)) {

                    uint8_t pixels[64], *c_ptr, *c_ptr1, *ptr;

                    int y, linesize;

                    /* NOTE: at end of line, the macroblock is handled as 420 */

                    idct_put(pixels, 8, block);

                    linesize = s->linesize[6 - j];

                    c_ptr = s->current_picture[6 - j] + c_offset;

                    ptr = pixels;

                    for(y = 0;y < 8; y++) {

                        /* convert to 411P */

                        c_ptr1 = c_ptr + linesize;

                        c_ptr1[0] = c_ptr[0] = (ptr[0] + ptr[1]) >> 1;

                        c_ptr1[1] = c_ptr[1] = (ptr[2] + ptr[3]) >> 1;

                        c_ptr1[2] = c_ptr[2] = (ptr[4] + ptr[5]) >> 1;

                        c_ptr1[3] = c_ptr[3] = (ptr[6] + ptr[7]) >> 1;

                        c_ptr += linesize * 2;

                        ptr += 8;

                    }

                } else {

                    /* don't ask me why they inverted Cb and Cr ! */

                    idct_put(s->current_picture[6 - j] + c_offset, 

                             s->linesize[6 - j], block);

                }

            }

            block += 64;

            mb++;

        }

    }

}

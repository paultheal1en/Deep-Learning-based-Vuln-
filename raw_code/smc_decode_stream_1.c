static void smc_decode_stream(SmcContext *s)

{

    int width = s->avctx->width;

    int height = s->avctx->height;

    int stride = s->frame.linesize[0];

    int i;

    int stream_ptr = 0;

    int chunk_size;

    unsigned char opcode;

    int n_blocks;

    unsigned int color_flags;

    unsigned int color_flags_a;

    unsigned int color_flags_b;

    unsigned int flag_mask;



    unsigned char *pixels = s->frame.data[0];



    int image_size = height * s->frame.linesize[0];

    int row_ptr = 0;

    int pixel_ptr = 0;

    int pixel_x, pixel_y;

    int row_inc = stride - 4;

    int block_ptr;

    int prev_block_ptr;

    int prev_block_ptr1, prev_block_ptr2;

    int prev_block_flag;

    int total_blocks;

    int color_table_index;  /* indexes to color pair, quad, or octet tables */

    int pixel;



    int color_pair_index = 0;

    int color_quad_index = 0;

    int color_octet_index = 0;



    /* make the palette available */

    memcpy(s->frame.data[1], s->pal, AVPALETTE_SIZE);



    chunk_size = AV_RB32(&s->buf[stream_ptr]) & 0x00FFFFFF;

    stream_ptr += 4;

    if (chunk_size != s->size)

        av_log(s->avctx, AV_LOG_INFO, "warning: MOV chunk size != encoded chunk size (%d != %d); using MOV chunk size\n",

            chunk_size, s->size);



    chunk_size = s->size;

    total_blocks = ((s->avctx->width + 3) / 4) * ((s->avctx->height + 3) / 4);



    /* traverse through the blocks */

    while (total_blocks) {

        /* sanity checks */

        /* make sure stream ptr hasn't gone out of bounds */

        if (stream_ptr > chunk_size) {

            av_log(s->avctx, AV_LOG_INFO, "SMC decoder just went out of bounds (stream ptr = %d, chunk size = %d)\n",

                stream_ptr, chunk_size);

            return;

        }

        /* make sure the row pointer hasn't gone wild */

        if (row_ptr >= image_size) {

            av_log(s->avctx, AV_LOG_INFO, "SMC decoder just went out of bounds (row ptr = %d, height = %d)\n",

                row_ptr, image_size);

            return;

        }



        opcode = s->buf[stream_ptr++];

        switch (opcode & 0xF0) {

        /* skip n blocks */

        case 0x00:

        case 0x10:

            n_blocks = GET_BLOCK_COUNT();

            while (n_blocks--) {

                ADVANCE_BLOCK();

            }

            break;



        /* repeat last block n times */

        case 0x20:

        case 0x30:

            n_blocks = GET_BLOCK_COUNT();



            /* sanity check */

            if ((row_ptr == 0) && (pixel_ptr == 0)) {

                av_log(s->avctx, AV_LOG_INFO, "encountered repeat block opcode (%02X) but no blocks rendered yet\n",

                    opcode & 0xF0);

                break;

            }



            /* figure out where the previous block started */

            if (pixel_ptr == 0)

                prev_block_ptr1 =

                    (row_ptr - s->avctx->width * 4) + s->avctx->width - 4;

            else

                prev_block_ptr1 = row_ptr + pixel_ptr - 4;



            while (n_blocks--) {

                block_ptr = row_ptr + pixel_ptr;

                prev_block_ptr = prev_block_ptr1;

                for (pixel_y = 0; pixel_y < 4; pixel_y++) {

                    for (pixel_x = 0; pixel_x < 4; pixel_x++) {

                        pixels[block_ptr++] = pixels[prev_block_ptr++];

                    }

                    block_ptr += row_inc;

                    prev_block_ptr += row_inc;

                }

                ADVANCE_BLOCK();

            }

            break;



        /* repeat previous pair of blocks n times */

        case 0x40:

        case 0x50:

            n_blocks = GET_BLOCK_COUNT();

            n_blocks *= 2;



            /* sanity check */

            if ((row_ptr == 0) && (pixel_ptr < 2 * 4)) {

                av_log(s->avctx, AV_LOG_INFO, "encountered repeat block opcode (%02X) but not enough blocks rendered yet\n",

                    opcode & 0xF0);

                break;

            }



            /* figure out where the previous 2 blocks started */

            if (pixel_ptr == 0)

                prev_block_ptr1 = (row_ptr - s->avctx->width * 4) +

                    s->avctx->width - 4 * 2;

            else if (pixel_ptr == 4)

                prev_block_ptr1 = (row_ptr - s->avctx->width * 4) + row_inc;

            else

                prev_block_ptr1 = row_ptr + pixel_ptr - 4 * 2;



            if (pixel_ptr == 0)

                prev_block_ptr2 = (row_ptr - s->avctx->width * 4) + row_inc;

            else

                prev_block_ptr2 = row_ptr + pixel_ptr - 4;



            prev_block_flag = 0;

            while (n_blocks--) {

                block_ptr = row_ptr + pixel_ptr;

                if (prev_block_flag)

                    prev_block_ptr = prev_block_ptr2;

                else

                    prev_block_ptr = prev_block_ptr1;

                prev_block_flag = !prev_block_flag;



                for (pixel_y = 0; pixel_y < 4; pixel_y++) {

                    for (pixel_x = 0; pixel_x < 4; pixel_x++) {

                        pixels[block_ptr++] = pixels[prev_block_ptr++];

                    }

                    block_ptr += row_inc;

                    prev_block_ptr += row_inc;

                }

                ADVANCE_BLOCK();

            }

            break;



        /* 1-color block encoding */

        case 0x60:

        case 0x70:

            n_blocks = GET_BLOCK_COUNT();

            pixel = s->buf[stream_ptr++];



            while (n_blocks--) {

                block_ptr = row_ptr + pixel_ptr;

                for (pixel_y = 0; pixel_y < 4; pixel_y++) {

                    for (pixel_x = 0; pixel_x < 4; pixel_x++) {

                        pixels[block_ptr++] = pixel;

                    }

                    block_ptr += row_inc;

                }

                ADVANCE_BLOCK();

            }

            break;



        /* 2-color block encoding */

        case 0x80:

        case 0x90:

            n_blocks = (opcode & 0x0F) + 1;



            /* figure out which color pair to use to paint the 2-color block */

            if ((opcode & 0xF0) == 0x80) {

                /* fetch the next 2 colors from bytestream and store in next

                 * available entry in the color pair table */

                for (i = 0; i < CPAIR; i++) {

                    pixel = s->buf[stream_ptr++];

                    color_table_index = CPAIR * color_pair_index + i;

                    s->color_pairs[color_table_index] = pixel;

                }

                /* this is the base index to use for this block */

                color_table_index = CPAIR * color_pair_index;

                color_pair_index++;

                /* wraparound */

                if (color_pair_index == COLORS_PER_TABLE)

                    color_pair_index = 0;

            } else

                color_table_index = CPAIR * s->buf[stream_ptr++];



            while (n_blocks--) {

                color_flags = AV_RB16(&s->buf[stream_ptr]);

                stream_ptr += 2;

                flag_mask = 0x8000;

                block_ptr = row_ptr + pixel_ptr;

                for (pixel_y = 0; pixel_y < 4; pixel_y++) {

                    for (pixel_x = 0; pixel_x < 4; pixel_x++) {

                        if (color_flags & flag_mask)

                            pixel = color_table_index + 1;

                        else

                            pixel = color_table_index;

                        flag_mask >>= 1;

                        pixels[block_ptr++] = s->color_pairs[pixel];

                    }

                    block_ptr += row_inc;

                }

                ADVANCE_BLOCK();

            }

            break;



        /* 4-color block encoding */

        case 0xA0:

        case 0xB0:

            n_blocks = (opcode & 0x0F) + 1;



            /* figure out which color quad to use to paint the 4-color block */

            if ((opcode & 0xF0) == 0xA0) {

                /* fetch the next 4 colors from bytestream and store in next

                 * available entry in the color quad table */

                for (i = 0; i < CQUAD; i++) {

                    pixel = s->buf[stream_ptr++];

                    color_table_index = CQUAD * color_quad_index + i;

                    s->color_quads[color_table_index] = pixel;

                }

                /* this is the base index to use for this block */

                color_table_index = CQUAD * color_quad_index;

                color_quad_index++;

                /* wraparound */

                if (color_quad_index == COLORS_PER_TABLE)

                    color_quad_index = 0;

            } else

                color_table_index = CQUAD * s->buf[stream_ptr++];



            while (n_blocks--) {

                color_flags = AV_RB32(&s->buf[stream_ptr]);

                stream_ptr += 4;

                /* flag mask actually acts as a bit shift count here */

                flag_mask = 30;

                block_ptr = row_ptr + pixel_ptr;

                for (pixel_y = 0; pixel_y < 4; pixel_y++) {

                    for (pixel_x = 0; pixel_x < 4; pixel_x++) {

                        pixel = color_table_index +

                            ((color_flags >> flag_mask) & 0x03);

                        flag_mask -= 2;

                        pixels[block_ptr++] = s->color_quads[pixel];

                    }

                    block_ptr += row_inc;

                }

                ADVANCE_BLOCK();

            }

            break;



        /* 8-color block encoding */

        case 0xC0:

        case 0xD0:

            n_blocks = (opcode & 0x0F) + 1;



            /* figure out which color octet to use to paint the 8-color block */

            if ((opcode & 0xF0) == 0xC0) {

                /* fetch the next 8 colors from bytestream and store in next

                 * available entry in the color octet table */

                for (i = 0; i < COCTET; i++) {

                    pixel = s->buf[stream_ptr++];

                    color_table_index = COCTET * color_octet_index + i;

                    s->color_octets[color_table_index] = pixel;

                }

                /* this is the base index to use for this block */

                color_table_index = COCTET * color_octet_index;

                color_octet_index++;

                /* wraparound */

                if (color_octet_index == COLORS_PER_TABLE)

                    color_octet_index = 0;

            } else

                color_table_index = COCTET * s->buf[stream_ptr++];



            while (n_blocks--) {

                /*

                  For this input of 6 hex bytes:

                    01 23 45 67 89 AB

                  Mangle it to this output:

                    flags_a = xx012456, flags_b = xx89A37B

                */

                /* build the color flags */

                color_flags_a =

                    ((AV_RB16(s->buf + stream_ptr    ) & 0xFFF0) << 8) |

                     (AV_RB16(s->buf + stream_ptr + 2) >> 4);

                color_flags_b =

                    ((AV_RB16(s->buf + stream_ptr + 4) & 0xFFF0) << 8) |

                    ((s->buf[stream_ptr + 1] & 0x0F) << 8) |

                    ((s->buf[stream_ptr + 3] & 0x0F) << 4) |

                    (s->buf[stream_ptr + 5] & 0x0F);

                stream_ptr += 6;



                color_flags = color_flags_a;

                /* flag mask actually acts as a bit shift count here */

                flag_mask = 21;

                block_ptr = row_ptr + pixel_ptr;

                for (pixel_y = 0; pixel_y < 4; pixel_y++) {

                    /* reload flags at third row (iteration pixel_y == 2) */

                    if (pixel_y == 2) {

                        color_flags = color_flags_b;

                        flag_mask = 21;

                    }

                    for (pixel_x = 0; pixel_x < 4; pixel_x++) {

                        pixel = color_table_index +

                            ((color_flags >> flag_mask) & 0x07);

                        flag_mask -= 3;

                        pixels[block_ptr++] = s->color_octets[pixel];

                    }

                    block_ptr += row_inc;

                }

                ADVANCE_BLOCK();

            }

            break;



        /* 16-color block encoding (every pixel is a different color) */

        case 0xE0:

            n_blocks = (opcode & 0x0F) + 1;



            while (n_blocks--) {

                block_ptr = row_ptr + pixel_ptr;

                for (pixel_y = 0; pixel_y < 4; pixel_y++) {

                    for (pixel_x = 0; pixel_x < 4; pixel_x++) {

                        pixels[block_ptr++] = s->buf[stream_ptr++];

                    }

                    block_ptr += row_inc;

                }

                ADVANCE_BLOCK();

            }

            break;



        case 0xF0:

            av_log(s->avctx, AV_LOG_INFO, "0xF0 opcode seen in SMC chunk (contact the developers)\n");

            break;

        }

    }

}

static void msvideo1_decode_8bit(Msvideo1Context *s)

{

    int block_ptr, pixel_ptr;

    int total_blocks;

    int pixel_x, pixel_y;  /* pixel width and height iterators */

    int block_x, block_y;  /* block width and height iterators */

    int blocks_wide, blocks_high;  /* width and height in 4x4 blocks */

    int block_inc;

    int row_dec;



    /* decoding parameters */

    int stream_ptr;

    unsigned char byte_a, byte_b;

    unsigned short flags;

    int skip_blocks;

    unsigned char colors[8];

    unsigned char *pixels = s->frame.data[0];

    unsigned char *prev_pixels = s->prev_frame.data[0];

    int stride = s->frame.linesize[0];



    stream_ptr = 0;

    skip_blocks = 0;

    blocks_wide = s->avctx->width / 4;

    blocks_high = s->avctx->height / 4;

    total_blocks = blocks_wide * blocks_high;

    block_inc = 4;

    row_dec = stride + 4;



    for (block_y = blocks_high; block_y > 0; block_y--) {

        block_ptr = ((block_y * 4) - 1) * stride;

        for (block_x = blocks_wide; block_x > 0; block_x--) {

            /* check if this block should be skipped */

            if (skip_blocks) {

                COPY_PREV_BLOCK();

                block_ptr += block_inc;

                skip_blocks--;

                total_blocks--;

                continue;

            }



            pixel_ptr = block_ptr;



            /* get the next two bytes in the encoded data stream */

            CHECK_STREAM_PTR(2);

            byte_a = s->buf[stream_ptr++];

            byte_b = s->buf[stream_ptr++];



            /* check if the decode is finished */

            if ((byte_a == 0) && (byte_b == 0) && (total_blocks == 0))

                return;

            else if ((byte_b & 0xFC) == 0x84) {

                /* skip code, but don't count the current block */

                skip_blocks = ((byte_b - 0x84) << 8) + byte_a - 1;

                COPY_PREV_BLOCK();

            } else if (byte_b < 0x80) {

                /* 2-color encoding */

                flags = (byte_b << 8) | byte_a;



                CHECK_STREAM_PTR(2);

                colors[0] = s->buf[stream_ptr++];

                colors[1] = s->buf[stream_ptr++];



                for (pixel_y = 0; pixel_y < 4; pixel_y++) {

                    for (pixel_x = 0; pixel_x < 4; pixel_x++, flags >>= 1)

                        pixels[pixel_ptr++] = colors[(flags & 0x1) ^ 1];

                    pixel_ptr -= row_dec;

                }

            } else if (byte_b >= 0x90) {

                /* 8-color encoding */

                flags = (byte_b << 8) | byte_a;



                CHECK_STREAM_PTR(8);

                memcpy(colors, &s->buf[stream_ptr], 8);

                stream_ptr += 8;



                for (pixel_y = 0; pixel_y < 4; pixel_y++) {

                    for (pixel_x = 0; pixel_x < 4; pixel_x++, flags >>= 1)

                        pixels[pixel_ptr++] = 

                            colors[((pixel_y & 0x2) << 1) + 

                                (pixel_x & 0x2) + ((flags & 0x1) ^ 1)];

                    pixel_ptr -= row_dec;

                }

            } else {

                /* 1-color encoding */

                colors[0] = byte_a;



                for (pixel_y = 0; pixel_y < 4; pixel_y++) {

                    for (pixel_x = 0; pixel_x < 4; pixel_x++)

                        pixels[pixel_ptr++] = colors[0];

                    pixel_ptr -= row_dec;

                }

            }



            block_ptr += block_inc;

            total_blocks--;

        }

    }



    /* make the palette available on the way out */

    if (s->avctx->pix_fmt == PIX_FMT_PAL8)

        memcpy(s->frame.data[1], s->palette, PALETTE_COUNT * 4);

}

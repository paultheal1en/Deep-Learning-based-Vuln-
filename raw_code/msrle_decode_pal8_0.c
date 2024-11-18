static void msrle_decode_pal8(MsrleContext *s)

{

    int stream_ptr = 0;

    unsigned char rle_code;

    unsigned char extra_byte;

    unsigned char stream_byte;

    int pixel_ptr = 0;

    int row_dec = s->frame.linesize[0];

    int row_ptr = (s->avctx->height - 1) * row_dec;

    int frame_size = row_dec * s->avctx->height;



    while (row_ptr >= 0) {

        FETCH_NEXT_STREAM_BYTE();

        rle_code = stream_byte;

        if (rle_code == 0) {

            /* fetch the next byte to see how to handle escape code */

            FETCH_NEXT_STREAM_BYTE();

            if (stream_byte == 0) {

                /* line is done, goto the next one */

                row_ptr -= row_dec;

                pixel_ptr = 0;

            } else if (stream_byte == 1) {

                /* decode is done */

                return;

            } else if (stream_byte == 2) {

                /* reposition frame decode coordinates */

                FETCH_NEXT_STREAM_BYTE();

                pixel_ptr += stream_byte;

                FETCH_NEXT_STREAM_BYTE();

                row_ptr -= stream_byte * row_dec;

            } else {

                /* copy pixels from encoded stream */

                if ((row_ptr + pixel_ptr + stream_byte > frame_size) ||

                    (row_ptr < 0)) {

                    printf(" MS RLE: frame ptr just went out of bounds (1)\n");

                    return;

                }



                rle_code = stream_byte;

                extra_byte = stream_byte & 0x01;

                if (stream_ptr + rle_code + extra_byte > s->size) {

                    printf(" MS RLE: stream ptr just went out of bounds (2)\n");

                    return;

                }



                while (rle_code--) {

                    FETCH_NEXT_STREAM_BYTE();

                    s->frame.data[0][row_ptr + pixel_ptr] = stream_byte;

                    pixel_ptr++;

                }



                /* if the RLE code is odd, skip a byte in the stream */

                if (extra_byte)

                    stream_ptr++;

            }

        } else {

            /* decode a run of data */

            if ((row_ptr + pixel_ptr + stream_byte > frame_size) ||

                (row_ptr < 0)) {

                printf(" MS RLE: frame ptr just went out of bounds (2)\n");

                return;

            }



            FETCH_NEXT_STREAM_BYTE();



            while(rle_code--) {

                s->frame.data[0][row_ptr + pixel_ptr] = stream_byte;

                pixel_ptr++;

            }

        }

    }



    /* make the palette available */

    memcpy(s->frame.data[1], s->palette, 256 * 4);



    /* one last sanity check on the way out */

    if (stream_ptr < s->size)

        printf(" MS RLE: ended frame decode with bytes left over (%d < %d)\n",

            stream_ptr, s->size);

}

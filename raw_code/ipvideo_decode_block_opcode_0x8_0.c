static int ipvideo_decode_block_opcode_0x8(IpvideoContext *s)

{

    int x, y;

    unsigned char P[2];

    unsigned int flags = 0;



    /* 2-color encoding for each 4x4 quadrant, or 2-color encoding on

     * either top and bottom or left and right halves */

    CHECK_STREAM_PTR(2);



    P[0] = *s->stream_ptr++;

    P[1] = *s->stream_ptr++;



    if (P[0] <= P[1]) {



        CHECK_STREAM_PTR(14);

        s->stream_ptr -= 2;



        for (y = 0; y < 16; y++) {

            // new values for each 4x4 block

            if (!(y & 3)) {

                P[0] = *s->stream_ptr++; P[1] = *s->stream_ptr++;

                flags = bytestream_get_le16(&s->stream_ptr);

            }



            for (x = 0; x < 4; x++, flags >>= 1)

                *s->pixel_ptr++ = P[flags & 1];

            s->pixel_ptr += s->stride - 4;

            // switch to right half

            if (y == 7) s->pixel_ptr -= 8 * s->stride - 4;

        }



    } else {



        /* need 10 more bytes */

        CHECK_STREAM_PTR(10);



        if (s->stream_ptr[4] <= s->stream_ptr[5]) {



            flags = bytestream_get_le32(&s->stream_ptr);



            /* vertical split; left & right halves are 2-color encoded */



            for (y = 0; y < 16; y++) {

                for (x = 0; x < 4; x++, flags >>= 1)

                    *s->pixel_ptr++ = P[flags & 1];

                s->pixel_ptr += s->stride - 4;

                // switch to right half

                if (y == 7) {

                    s->pixel_ptr -= 8 * s->stride - 4;

                    P[0] = *s->stream_ptr++; P[1] = *s->stream_ptr++;

                    flags = bytestream_get_le32(&s->stream_ptr);

                }

            }



        } else {



            /* horizontal split; top & bottom halves are 2-color encoded */



            for (y = 0; y < 8; y++) {

                if (y == 4) {

                    P[0] = *s->stream_ptr++;

                    P[1] = *s->stream_ptr++;

                }

                flags = *s->stream_ptr++ | 0x100;



                for (; flags != 1; flags >>= 1)

                    *s->pixel_ptr++ = P[flags & 1];

                s->pixel_ptr += s->line_inc;

            }

        }

    }



    /* report success */

    return 0;

}

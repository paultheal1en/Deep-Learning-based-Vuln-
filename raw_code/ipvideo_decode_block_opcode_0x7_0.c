static int ipvideo_decode_block_opcode_0x7(IpvideoContext *s)

{

    int x, y;

    unsigned char P[2];

    unsigned int flags;



    /* 2-color encoding */

    CHECK_STREAM_PTR(2);



    P[0] = *s->stream_ptr++;

    P[1] = *s->stream_ptr++;



    if (P[0] <= P[1]) {



        /* need 8 more bytes from the stream */

        CHECK_STREAM_PTR(8);



        for (y = 0; y < 8; y++) {

            flags = *s->stream_ptr++ | 0x100;

            for (; flags != 1; flags >>= 1)

                *s->pixel_ptr++ = P[flags & 1];

            s->pixel_ptr += s->line_inc;

        }



    } else {



        /* need 2 more bytes from the stream */

        CHECK_STREAM_PTR(2);



        flags = bytestream_get_le16(&s->stream_ptr);

        for (y = 0; y < 8; y += 2) {

            for (x = 0; x < 8; x += 2, flags >>= 1) {

                s->pixel_ptr[x                ] =

                s->pixel_ptr[x + 1            ] =

                s->pixel_ptr[x +     s->stride] =

                s->pixel_ptr[x + 1 + s->stride] = P[flags & 1];

            }

            s->pixel_ptr += s->stride * 2;

        }

    }



    /* report success */

    return 0;

}

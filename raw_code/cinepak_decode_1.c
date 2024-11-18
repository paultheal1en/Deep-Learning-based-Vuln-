static int cinepak_decode (CinepakContext *s)

{

    const uint8_t  *eod = (s->data + s->size);

    int           i, result, strip_size, frame_flags, num_strips;

    int           y0 = 0;

    int           encoded_buf_size;



    if (s->size < 10)

        return -1;



    frame_flags = s->data[0];

    num_strips  = AV_RB16 (&s->data[8]);

    encoded_buf_size = ((s->data[1] << 16) | AV_RB16 (&s->data[2]));



    /* if this is the first frame, check for deviant Sega FILM data */

    if (s->sega_film_skip_bytes == -1) {

        if (encoded_buf_size != s->size) {

            /* If the encoded frame size differs from the frame size as indicated

             * by the container file, this data likely comes from a Sega FILM/CPK file.

             * If the frame header is followed by the bytes FE 00 00 06 00 00 then

             * this is probably one of the two known files that have 6 extra bytes

             * after the frame header. Else, assume 2 extra bytes. */

            if ((s->data[10] == 0xFE) &&

                (s->data[11] == 0x00) &&

                (s->data[12] == 0x00) &&

                (s->data[13] == 0x06) &&

                (s->data[14] == 0x00) &&

                (s->data[15] == 0x00))

                s->sega_film_skip_bytes = 6;

            else

                s->sega_film_skip_bytes = 2;

        } else

            s->sega_film_skip_bytes = 0;

    }



    s->data += 10 + s->sega_film_skip_bytes;



    if (num_strips > MAX_STRIPS)

        num_strips = MAX_STRIPS;



    for (i=0; i < num_strips; i++) {

        if ((s->data + 12) > eod)

            return -1;



        s->strips[i].id = s->data[0];

        s->strips[i].y1 = y0;

        s->strips[i].x1 = 0;

        s->strips[i].y2 = y0 + AV_RB16 (&s->data[8]);

        s->strips[i].x2 = s->avctx->width;



        strip_size = AV_RB24 (&s->data[1]) - 12;

        s->data   += 12;

        strip_size = ((s->data + strip_size) > eod) ? (eod - s->data) : strip_size;



        if ((i > 0) && !(frame_flags & 0x01)) {

            memcpy (s->strips[i].v4_codebook, s->strips[i-1].v4_codebook,

                sizeof(s->strips[i].v4_codebook));

            memcpy (s->strips[i].v1_codebook, s->strips[i-1].v1_codebook,

                sizeof(s->strips[i].v1_codebook));

        }



        result = cinepak_decode_strip (s, &s->strips[i], s->data, strip_size);



        if (result != 0)

            return result;



        s->data += strip_size;

        y0    = s->strips[i].y2;

    }

    return 0;

}

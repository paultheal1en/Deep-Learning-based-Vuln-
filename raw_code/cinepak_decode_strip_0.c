static int cinepak_decode_strip (CinepakContext *s,

                                 cvid_strip *strip, const uint8_t *data, int size)

{

    const uint8_t *eod = (data + size);

    int      chunk_id, chunk_size;



    /* coordinate sanity checks */

    if (strip->x1 >= s->width  || strip->x2 > s->width  ||

        strip->y1 >= s->height || strip->y2 > s->height ||

        strip->x1 >= strip->x2 || strip->y1 >= strip->y2)

        return -1;



    while ((data + 4) <= eod) {

        chunk_id   = data[0];

        chunk_size = AV_RB24 (&data[1]) - 4;

        if(chunk_size < 0)

            return -1;



        data      += 4;

        chunk_size = ((data + chunk_size) > eod) ? (eod - data) : chunk_size;



        switch (chunk_id) {



        case 0x20:

        case 0x21:

        case 0x24:

        case 0x25:

            cinepak_decode_codebook (strip->v4_codebook, chunk_id,

                chunk_size, data);

            break;



        case 0x22:

        case 0x23:

        case 0x26:

        case 0x27:

            cinepak_decode_codebook (strip->v1_codebook, chunk_id,

                chunk_size, data);

            break;



        case 0x30:

        case 0x31:

        case 0x32:

            return cinepak_decode_vectors (s, strip, chunk_id,

                chunk_size, data);

        }



        data += chunk_size;

    }



    return -1;

}

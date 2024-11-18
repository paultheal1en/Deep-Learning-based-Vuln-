static int read_rle_sgi(uint8_t *out_buf, SgiState *s)

{

    uint8_t *dest_row;

    unsigned int len = s->height * s->depth * 4;

    GetByteContext g_table = s->g;

    unsigned int y, z;

    unsigned int start_offset;

    int linesize, ret;



    /* size of  RLE offset and length tables */

    if (len * 2 > bytestream2_get_bytes_left(&s->g)) {

        return AVERROR_INVALIDDATA;

    }



    for (z = 0; z < s->depth; z++) {

        dest_row = out_buf;

        for (y = 0; y < s->height; y++) {

            linesize = s->width * s->depth * s->bytes_per_channel;

            dest_row -= s->linesize;

            start_offset = bytestream2_get_be32(&g_table);

            bytestream2_seek(&s->g, start_offset, SEEK_SET);

            if (s->bytes_per_channel == 1)

                ret = expand_rle_row8(s, dest_row + z, linesize, s->depth);

            else

                ret = expand_rle_row16(s, (uint16_t *)dest_row + z, linesize, s->depth);

            if (ret != s->width)

                return AVERROR_INVALIDDATA;

        }

    }

    return 0;

}

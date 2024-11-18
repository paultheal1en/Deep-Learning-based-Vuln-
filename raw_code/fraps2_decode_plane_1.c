static int fraps2_decode_plane(FrapsContext *s, uint8_t *dst, int stride, int w,

                               int h, const uint8_t *src, int size, int Uoff,

                               const int step)

{

    int i, j;

    GetBitContext gb;

    VLC vlc;

    Node nodes[512];



    for(i = 0; i < 256; i++)

        nodes[i].count = bytestream_get_le32(&src);

    size -= 1024;

    if (ff_huff_build_tree(s->avctx, &vlc, 256, nodes, huff_cmp,

                           FF_HUFFMAN_FLAG_ZERO_COUNT) < 0)

        return -1;

    /* we have built Huffman table and are ready to decode plane */



    /* convert bits so they may be used by standard bitreader */

    s->dsp.bswap_buf((uint32_t *)s->tmpbuf, (const uint32_t *)src, size >> 2);



    init_get_bits(&gb, s->tmpbuf, size * 8);

    for(j = 0; j < h; j++){

        for(i = 0; i < w*step; i += step){

            dst[i] = get_vlc2(&gb, vlc.table, 9, 3);

            /* lines are stored as deltas between previous lines

             * and we need to add 0x80 to the first lines of chroma planes

             */

            if(j) dst[i] += dst[i - stride];

            else if(Uoff) dst[i] += 0x80;

        }

        dst += stride;

        if(get_bits_left(&gb) < 0){

            free_vlc(&vlc);

            return -1;

        }

    }

    free_vlc(&vlc);

    return 0;

}

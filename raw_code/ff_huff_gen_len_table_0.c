int ff_huff_gen_len_table(uint8_t *dst, const uint64_t *stats, int stats_size, int skip0)

{

    HeapElem *h  = av_malloc_array(sizeof(*h), stats_size);

    int *up      = av_malloc_array(sizeof(*up) * 2, stats_size);

    uint8_t *len = av_malloc_array(sizeof(*len) * 2, stats_size);

    uint16_t *map= av_malloc_array(sizeof(*map), stats_size);

    int offset, i, next;

    int size = 0;

    int ret = 0;



    if (!h || !up || !len) {

        ret = AVERROR(ENOMEM);

        goto end;

    }



    for (i = 0; i<stats_size; i++) {

        dst[i] = 255;

        if (stats[i] || !skip0)

            map[size++] = i;

    }



    for (offset = 1; ; offset <<= 1) {

        for (i=0; i < size; i++) {

            h[i].name = i;

            h[i].val = (stats[map[i]] << 14) + offset;

        }

        for (i = size / 2 - 1; i >= 0; i--)

            heap_sift(h, i, size);



        for (next = size; next < size * 2 - 1; next++) {

            // merge the two smallest entries, and put it back in the heap

            uint64_t min1v = h[0].val;

            up[h[0].name] = next;

            h[0].val = INT64_MAX;

            heap_sift(h, 0, size);

            up[h[0].name] = next;

            h[0].name = next;

            h[0].val += min1v;

            heap_sift(h, 0, size);

        }



        len[2 * size - 2] = 0;

        for (i = 2 * size - 3; i >= size; i--)

            len[i] = len[up[i]] + 1;

        for (i = 0; i < size; i++) {

            dst[map[i]] = len[up[i]] + 1;

            if (dst[map[i]] >= 32) break;

        }

        if (i==size) break;

    }

end:

    av_free(h);

    av_free(up);

    av_free(len);

    av_free(map);

    return ret;

}

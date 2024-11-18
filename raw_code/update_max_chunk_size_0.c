static void update_max_chunk_size(BDRVDMGState *s, uint32_t chunk,

                                  uint32_t *max_compressed_size,

                                  uint32_t *max_sectors_per_chunk)

{

    uint32_t compressed_size = 0;

    uint32_t uncompressed_sectors = 0;



    switch (s->types[chunk]) {

    case 0x80000005: /* zlib compressed */

    case 0x80000006: /* bzip2 compressed */

        compressed_size = s->lengths[chunk];

        uncompressed_sectors = s->sectorcounts[chunk];

        break;

    case 1: /* copy */

        uncompressed_sectors = (s->lengths[chunk] + 511) / 512;

        break;

    case 2: /* zero */

        uncompressed_sectors = s->sectorcounts[chunk];

        break;

    }



    if (compressed_size > *max_compressed_size) {

        *max_compressed_size = compressed_size;

    }

    if (uncompressed_sectors > *max_sectors_per_chunk) {

        *max_sectors_per_chunk = uncompressed_sectors;

    }

}

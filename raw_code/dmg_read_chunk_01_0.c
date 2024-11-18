static inline int dmg_read_chunk(BlockDriverState *bs, uint64_t sector_num)

{

    BDRVDMGState *s = bs->opaque;



    if (!is_sector_in_chunk(s, s->current_chunk, sector_num)) {

        int ret;

        uint32_t chunk = search_chunk(s, sector_num);

#ifdef CONFIG_BZIP2

        uint64_t total_out;

#endif



        if (chunk >= s->n_chunks) {

            return -1;

        }



        s->current_chunk = s->n_chunks;

        switch (s->types[chunk]) { /* block entry type */

        case 0x80000005: { /* zlib compressed */

            /* we need to buffer, because only the chunk as whole can be

             * inflated. */

            ret = bdrv_pread(bs->file, s->offsets[chunk],

                             s->compressed_chunk, s->lengths[chunk]);

            if (ret != s->lengths[chunk]) {

                return -1;

            }



            s->zstream.next_in = s->compressed_chunk;

            s->zstream.avail_in = s->lengths[chunk];

            s->zstream.next_out = s->uncompressed_chunk;

            s->zstream.avail_out = 512 * s->sectorcounts[chunk];

            ret = inflateReset(&s->zstream);

            if (ret != Z_OK) {

                return -1;

            }

            ret = inflate(&s->zstream, Z_FINISH);

            if (ret != Z_STREAM_END ||

                s->zstream.total_out != 512 * s->sectorcounts[chunk]) {

                return -1;

            }

            break; }

#ifdef CONFIG_BZIP2

        case 0x80000006: /* bzip2 compressed */

            /* we need to buffer, because only the chunk as whole can be

             * inflated. */

            ret = bdrv_pread(bs->file, s->offsets[chunk],

                             s->compressed_chunk, s->lengths[chunk]);

            if (ret != s->lengths[chunk]) {

                return -1;

            }



            ret = BZ2_bzDecompressInit(&s->bzstream, 0, 0);

            if (ret != BZ_OK) {

                return -1;

            }

            s->bzstream.next_in = (char *)s->compressed_chunk;

            s->bzstream.avail_in = (unsigned int) s->lengths[chunk];

            s->bzstream.next_out = (char *)s->uncompressed_chunk;

            s->bzstream.avail_out = (unsigned int) 512 * s->sectorcounts[chunk];

            ret = BZ2_bzDecompress(&s->bzstream);

            total_out = ((uint64_t)s->bzstream.total_out_hi32 << 32) +

                        s->bzstream.total_out_lo32;

            BZ2_bzDecompressEnd(&s->bzstream);

            if (ret != BZ_STREAM_END ||

                total_out != 512 * s->sectorcounts[chunk]) {

                return -1;

            }

            break;

#endif /* CONFIG_BZIP2 */

        case 1: /* copy */

            ret = bdrv_pread(bs->file, s->offsets[chunk],

                             s->uncompressed_chunk, s->lengths[chunk]);

            if (ret != s->lengths[chunk]) {

                return -1;

            }

            break;

        case 2: /* zero */

            memset(s->uncompressed_chunk, 0, 512 * s->sectorcounts[chunk]);

            break;

        }

        s->current_chunk = chunk;

    }

    return 0;

}

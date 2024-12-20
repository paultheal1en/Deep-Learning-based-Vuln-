static int matroska_decode_buffer(uint8_t** buf, int* buf_size,

                                  MatroskaTrack *track)

{

    MatroskaTrackEncoding *encodings = track->encodings.elem;

    uint8_t* data = *buf;

    int isize = *buf_size;

    uint8_t* pkt_data = NULL;

    int pkt_size = isize;

    int result = 0;

    int olen;



    if (pkt_size >= 10000000)

        return -1;



    switch (encodings[0].compression.algo) {

    case MATROSKA_TRACK_ENCODING_COMP_HEADERSTRIP:

        return encodings[0].compression.settings.size;

    case MATROSKA_TRACK_ENCODING_COMP_LZO:

        do {

            olen = pkt_size *= 3;

            pkt_data = av_realloc(pkt_data, pkt_size+AV_LZO_OUTPUT_PADDING);

            result = av_lzo1x_decode(pkt_data, &olen, data, &isize);

        } while (result==AV_LZO_OUTPUT_FULL && pkt_size<10000000);

        if (result)

            goto failed;

        pkt_size -= olen;

        break;

#if CONFIG_ZLIB

    case MATROSKA_TRACK_ENCODING_COMP_ZLIB: {

        z_stream zstream = {0};

        if (inflateInit(&zstream) != Z_OK)

            return -1;

        zstream.next_in = data;

        zstream.avail_in = isize;

        do {

            pkt_size *= 3;

            pkt_data = av_realloc(pkt_data, pkt_size);

            zstream.avail_out = pkt_size - zstream.total_out;

            zstream.next_out = pkt_data + zstream.total_out;

            result = inflate(&zstream, Z_NO_FLUSH);

        } while (result==Z_OK && pkt_size<10000000);

        pkt_size = zstream.total_out;

        inflateEnd(&zstream);

        if (result != Z_STREAM_END)

            goto failed;

        break;

    }

#endif

#if CONFIG_BZLIB

    case MATROSKA_TRACK_ENCODING_COMP_BZLIB: {

        bz_stream bzstream = {0};

        if (BZ2_bzDecompressInit(&bzstream, 0, 0) != BZ_OK)

            return -1;

        bzstream.next_in = data;

        bzstream.avail_in = isize;

        do {

            pkt_size *= 3;

            pkt_data = av_realloc(pkt_data, pkt_size);

            bzstream.avail_out = pkt_size - bzstream.total_out_lo32;

            bzstream.next_out = pkt_data + bzstream.total_out_lo32;

            result = BZ2_bzDecompress(&bzstream);

        } while (result==BZ_OK && pkt_size<10000000);

        pkt_size = bzstream.total_out_lo32;

        BZ2_bzDecompressEnd(&bzstream);

        if (result != BZ_STREAM_END)

            goto failed;

        break;

    }

#endif

    default:

        return -1;

    }



    *buf = pkt_data;

    *buf_size = pkt_size;

    return 0;

 failed:

    av_free(pkt_data);

    return -1;

}

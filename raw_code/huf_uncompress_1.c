static int huf_uncompress(GetByteContext *gb,

                          uint16_t *dst, int dst_size)

{

    int32_t src_size, im, iM;

    uint32_t nBits;

    uint64_t *freq;

    HufDec *hdec;

    int ret, i;



    src_size = bytestream2_get_le32(gb);

    im = bytestream2_get_le32(gb);

    iM = bytestream2_get_le32(gb);

    bytestream2_skip(gb, 4);

    nBits = bytestream2_get_le32(gb);

    if (im < 0 || im >= HUF_ENCSIZE ||

        iM < 0 || iM >= HUF_ENCSIZE ||

        src_size < 0)

        return AVERROR_INVALIDDATA;



    bytestream2_skip(gb, 4);



    freq = av_calloc(HUF_ENCSIZE, sizeof(*freq));

    hdec = av_calloc(HUF_DECSIZE, sizeof(*hdec));

    if (!freq || !hdec) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    if ((ret = huf_unpack_enc_table(gb, im, iM, freq)) < 0)

        goto fail;



    if (nBits > 8 * bytestream2_get_bytes_left(gb)) {

        ret = AVERROR_INVALIDDATA;

        goto fail;

    }



    if ((ret = huf_build_dec_table(freq, im, iM, hdec)) < 0)

        goto fail;

    ret = huf_decode(freq, hdec, gb, nBits, iM, dst_size, dst);



fail:

    for (i = 0; i < HUF_DECSIZE; i++) {

        if (hdec[i].p)

            av_freep(&hdec[i].p);

    }



    av_free(freq);

    av_free(hdec);



    return ret;

}

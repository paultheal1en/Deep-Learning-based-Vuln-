int avpriv_exif_decode_ifd(void *logctx, GetByteContext *gbytes, int le,

                           int depth, AVDictionary **metadata)

{

    int i, ret;

    int entries;



    entries = ff_tget_short(gbytes, le);



    if (bytestream2_get_bytes_left(gbytes) < entries * 12) {

        return AVERROR_INVALIDDATA;

    }



    for (i = 0; i < entries; i++) {

        if ((ret = exif_decode_tag(logctx, gbytes, le, depth, metadata)) < 0) {

            return ret;

        }

    }



    // return next IDF offset or 0x000000000 or a value < 0 for failure

    return ff_tget_long(gbytes, le);

}

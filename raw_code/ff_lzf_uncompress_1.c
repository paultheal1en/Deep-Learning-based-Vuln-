int ff_lzf_uncompress(GetByteContext *gb, uint8_t **buf, int64_t *size)

{

    int ret     = 0;

    uint8_t *p  = *buf;

    int64_t len = 0;



    while (bytestream2_get_bytes_left(gb) > 2) {

        uint8_t s = bytestream2_get_byte(gb);



        if (s < LZF_LITERAL_MAX) {

            s++;

            if (s > *size - len) {

                *size += *size /2;

                ret = av_reallocp(buf, *size);

                if (ret < 0)

                    return ret;


            }



            bytestream2_get_buffer(gb, p, s);

            p   += s;

            len += s;

        } else {

            int l   = 2 + (s >> 5);

            int off = ((s & 0x1f) << 8) + 1;



            if (l == LZF_LONG_BACKREF)

                l += bytestream2_get_byte(gb);



            off += bytestream2_get_byte(gb);



            if (off > len)

                return AVERROR_INVALIDDATA;



            if (l > *size - len) {

                *size += *size / 2;

                ret = av_reallocp(buf, *size);

                if (ret < 0)

                    return ret;


            }



            av_memcpy_backptr(p, off, l);



            p   += l;

            len += l;

        }

    }



    *size = len;



    return 0;

}
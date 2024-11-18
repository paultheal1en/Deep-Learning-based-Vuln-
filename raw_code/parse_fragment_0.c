static int parse_fragment(AVFormatContext *s, const char *filename, int64_t *start_ts, int64_t *duration, int64_t *moof_size, int64_t size)

{

    AVIOContext *in;

    int ret;

    uint32_t len;

    if ((ret = avio_open2(&in, filename, AVIO_FLAG_READ, &s->interrupt_callback, NULL)) < 0)

        return ret;

    ret = AVERROR(EIO);

    *moof_size = avio_rb32(in);

    if (*moof_size < 8 || *moof_size > size)

        goto fail;

    if (avio_rl32(in) != MKTAG('m','o','o','f'))

        goto fail;

    len = avio_rb32(in);

    if (len > *moof_size)

        goto fail;

    if (avio_rl32(in) != MKTAG('m','f','h','d'))

        goto fail;

    avio_seek(in, len - 8, SEEK_CUR);

    avio_rb32(in); /* traf size */

    if (avio_rl32(in) != MKTAG('t','r','a','f'))

        goto fail;

    while (avio_tell(in) < *moof_size) {

        uint32_t len = avio_rb32(in);

        uint32_t tag = avio_rl32(in);

        int64_t end = avio_tell(in) + len - 8;

        if (len < 8 || len >= *moof_size)

            goto fail;

        if (tag == MKTAG('u','u','i','d')) {

            const uint8_t tfxd[] = {

                0x6d, 0x1d, 0x9b, 0x05, 0x42, 0xd5, 0x44, 0xe6,

                0x80, 0xe2, 0x14, 0x1d, 0xaf, 0xf7, 0x57, 0xb2

            };

            uint8_t uuid[16];

            avio_read(in, uuid, 16);

            if (!memcmp(uuid, tfxd, 16) && len >= 8 + 16 + 4 + 16) {

                avio_seek(in, 4, SEEK_CUR);

                *start_ts = avio_rb64(in);

                *duration = avio_rb64(in);

                ret = 0;

                break;

            }

        }

        avio_seek(in, end, SEEK_SET);

    }

fail:

    avio_close(in);

    return ret;

}

static int mov_read_replaygain(MOVContext *c, AVIOContext *pb, int size)

{

    int64_t end = avio_tell(pb) + size;

    uint8_t *key = NULL, *val = NULL;

    int i;



    for (i = 0; i < 2; i++) {

        uint8_t **p;

        uint32_t len, tag;



        if (end - avio_tell(pb) <= 12)

            break;



        len = avio_rb32(pb);

        tag = avio_rl32(pb);

        avio_skip(pb, 4); // flags



        if (len < 12 || len - 12 > end - avio_tell(pb))

            break;

        len -= 12;



        if (tag == MKTAG('n', 'a', 'm', 'e'))

            p = &key;

        else if (tag == MKTAG('d', 'a', 't', 'a') && len > 4) {

            avio_skip(pb, 4);

            len -= 4;

            p = &val;

        } else

            break;



        *p = av_malloc(len + 1);

        if (!*p)

            break;

        avio_read(pb, *p, len);

        (*p)[len] = 0;

    }



    if (key && val) {

        av_dict_set(&c->fc->metadata, key, val,

                    AV_DICT_DONT_STRDUP_KEY | AV_DICT_DONT_STRDUP_VAL);

        key = val = NULL;

    }



    avio_seek(pb, end, SEEK_SET);

    av_freep(&key);

    av_freep(&val);

    return 0;

}

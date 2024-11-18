int ff_vorbiscomment_write(uint8_t **p, AVDictionary **m,

                           const char *vendor_string)

{

    bytestream_put_le32(p, strlen(vendor_string));

    bytestream_put_buffer(p, vendor_string, strlen(vendor_string));

    if (*m) {

        int count = av_dict_count(*m);

        AVDictionaryEntry *tag = NULL;

        bytestream_put_le32(p, count);

        while ((tag = av_dict_get(*m, "", tag, AV_DICT_IGNORE_SUFFIX))) {

            unsigned int len1 = strlen(tag->key);

            unsigned int len2 = strlen(tag->value);

            bytestream_put_le32(p, len1+1+len2);

            bytestream_put_buffer(p, tag->key, len1);

            bytestream_put_byte(p, '=');

            bytestream_put_buffer(p, tag->value, len2);

        }

    } else

        bytestream_put_le32(p, 0);

    return 0;

}

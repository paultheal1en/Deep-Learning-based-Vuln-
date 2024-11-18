static int add_string_metadata(int count, const char *name,

                               TiffContext *s)

{

    char *value;



    if (bytestream2_get_bytes_left(&s->gb) < count)

        return AVERROR_INVALIDDATA;



    value = av_malloc(count + 1);

    if (!value)

        return AVERROR(ENOMEM);



    bytestream2_get_bufferu(&s->gb, value, count);

    value[count] = 0;



    av_dict_set(&s->picture.metadata, name, value, AV_DICT_DONT_STRDUP_VAL);

    return 0;

}

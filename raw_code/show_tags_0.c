static inline int show_tags(WriterContext *wctx, AVDictionary *tags, int section_id)

{

    AVDictionaryEntry *tag = NULL;

    int ret = 0;



    if (!tags)

        return 0;

    writer_print_section_header(wctx, section_id);



    while ((tag = av_dict_get(tags, "", tag, AV_DICT_IGNORE_SUFFIX))) {

        ret = writer_print_string(wctx, tag->key, tag->value, 0);

        if (ret < 0)

            break;

    }

    writer_print_section_footer(wctx);



    return ret;

}

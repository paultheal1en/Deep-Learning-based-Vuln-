static inline void show_tags(WriterContext *wctx, AVDictionary *tags, int section_id)

{

    AVDictionaryEntry *tag = NULL;



    if (!tags)

        return;

    writer_print_section_header(wctx, section_id);

    while ((tag = av_dict_get(tags, "", tag, AV_DICT_IGNORE_SUFFIX)))

        writer_print_string(wctx, tag->key, tag->value, 0);

    writer_print_section_footer(wctx);

}

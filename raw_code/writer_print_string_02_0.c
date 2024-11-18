static inline int writer_print_string(WriterContext *wctx,

                                      const char *key, const char *val, int opt)

{

    const struct section *section = wctx->section[wctx->level];

    int ret = 0;



    if (opt && !(wctx->writer->flags & WRITER_FLAG_DISPLAY_OPTIONAL_FIELDS))

        return 0;



    if (section->show_all_entries || av_dict_get(section->entries_to_show, key, NULL, 0)) {

        wctx->writer->print_string(wctx, key, val);

        wctx->nb_item[wctx->level]++;

    }



    return ret;

}

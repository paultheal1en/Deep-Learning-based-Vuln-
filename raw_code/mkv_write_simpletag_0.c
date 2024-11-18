static void mkv_write_simpletag(AVIOContext *pb, AVDictionaryEntry *t)

{

    uint8_t *key = av_strdup(t->key);

    uint8_t *p   = key;

    const uint8_t *lang = NULL;

    ebml_master tag;



    if ((p = strrchr(p, '-')) &&

        (lang = av_convert_lang_to(p + 1, AV_LANG_ISO639_2_BIBL)))

        *p = 0;



    p = key;

    while (*p) {

        if (*p == ' ')

            *p = '_';

        else if (*p >= 'a' && *p <= 'z')

            *p -= 'a' - 'A';

        p++;

    }



    tag = start_ebml_master(pb, MATROSKA_ID_SIMPLETAG, 0);

    put_ebml_string(pb, MATROSKA_ID_TAGNAME, key);

    if (lang)

        put_ebml_string(pb, MATROSKA_ID_TAGLANG, lang);

    put_ebml_string(pb, MATROSKA_ID_TAGSTRING, t->value);

    end_ebml_master(pb, tag);



    av_freep(&key);

}

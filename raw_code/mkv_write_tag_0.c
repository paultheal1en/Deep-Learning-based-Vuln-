static int mkv_write_tag(AVFormatContext *s, AVDictionary *m, unsigned int elementid,

                         unsigned int uid, ebml_master *tags)

{

    MatroskaMuxContext *mkv = s->priv_data;

    ebml_master tag, targets;

    AVDictionaryEntry *t = NULL;

    int ret;



    if (!tags->pos) {

        ret = mkv_add_seekhead_entry(mkv->main_seekhead, MATROSKA_ID_TAGS, avio_tell(s->pb));

        if (ret < 0) return ret;



        *tags = start_ebml_master(s->pb, MATROSKA_ID_TAGS, 0);

    }



    tag     = start_ebml_master(s->pb, MATROSKA_ID_TAG,        0);

    targets = start_ebml_master(s->pb, MATROSKA_ID_TAGTARGETS, 0);

    if (elementid)

        put_ebml_uint(s->pb, elementid, uid);

    end_ebml_master(s->pb, targets);



    while ((t = av_dict_get(m, "", t, AV_DICT_IGNORE_SUFFIX)))

        if (av_strcasecmp(t->key, "title") &&

            av_strcasecmp(t->key, "encoding_tool"))

            mkv_write_simpletag(s->pb, t);



    end_ebml_master(s->pb, tag);

    return 0;

}

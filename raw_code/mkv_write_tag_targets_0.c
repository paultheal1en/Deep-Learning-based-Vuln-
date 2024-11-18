static int mkv_write_tag_targets(AVFormatContext *s,

                                 unsigned int elementid, unsigned int uid,

                                 ebml_master *tags, ebml_master* tag)

{

    AVIOContext *pb;

    MatroskaMuxContext *mkv = s->priv_data;

    ebml_master targets;

    int ret;



    if (!tags->pos) {

        ret = mkv_add_seekhead_entry(mkv->main_seekhead, MATROSKA_ID_TAGS, avio_tell(s->pb));

        if (ret < 0) return ret;



        start_ebml_master_crc32(s->pb, &mkv->tags_bc, tags, MATROSKA_ID_TAGS, 0);

    }

    pb = mkv->tags_bc;



    *tag     = start_ebml_master(pb, MATROSKA_ID_TAG,       0);

    targets = start_ebml_master(pb, MATROSKA_ID_TAGTARGETS, 0);

    if (elementid)

        put_ebml_uint(pb, elementid, uid);

    end_ebml_master(pb, targets);

    return 0;

}

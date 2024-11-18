static int mov_write_ilst_tag(AVIOContext *pb, MOVMuxContext *mov,

                              AVFormatContext *s)

{

    int64_t pos = avio_tell(pb);

    avio_wb32(pb, 0); /* size */

    ffio_wfourcc(pb, "ilst");

    mov_write_string_metadata(s, pb, "\251nam", "title"    , 1);

    mov_write_string_metadata(s, pb, "\251ART", "artist"   , 1);

    mov_write_string_metadata(s, pb, "aART", "album_artist", 1);

    mov_write_string_metadata(s, pb, "\251wrt", "composer" , 1);

    mov_write_string_metadata(s, pb, "\251alb", "album"    , 1);

    mov_write_string_metadata(s, pb, "\251day", "date"     , 1);

    mov_write_string_tag(pb, "\251too", LIBAVFORMAT_IDENT, 0, 1);

    mov_write_string_metadata(s, pb, "\251cmt", "comment"  , 1);

    mov_write_string_metadata(s, pb, "\251gen", "genre"    , 1);

    mov_write_string_metadata(s, pb, "\251cpy", "copyright", 1);

    mov_write_string_metadata(s, pb, "\251grp", "grouping" , 1);

    mov_write_string_metadata(s, pb, "\251lyr", "lyrics"   , 1);

    mov_write_string_metadata(s, pb, "desc",    "description",1);

    mov_write_string_metadata(s, pb, "ldes",    "synopsis" , 1);

    mov_write_string_metadata(s, pb, "tvsh",    "show"     , 1);

    mov_write_string_metadata(s, pb, "tven",    "episode_id",1);

    mov_write_string_metadata(s, pb, "tvnn",    "network"  , 1);

    mov_write_trkn_tag(pb, mov, s);

    return update_size(pb, pos);

}

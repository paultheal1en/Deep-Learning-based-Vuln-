static int mov_write_tfhd_tag(AVIOContext *pb, MOVTrack *track,

                              int64_t moof_offset)

{

    int64_t pos = avio_tell(pb);

    uint32_t flags = MOV_TFHD_DEFAULT_SIZE | MOV_TFHD_DEFAULT_DURATION |

                     MOV_TFHD_BASE_DATA_OFFSET;

    if (!track->entry) {

        flags |= MOV_TFHD_DURATION_IS_EMPTY;

    } else {

        flags |= MOV_TFHD_DEFAULT_FLAGS;

    }



    /* Don't set a default sample size, the silverlight player refuses

     * to play files with that set. Don't set a default sample duration,

     * WMP freaks out if it is set. */

    if (track->mode == MODE_ISM)

        flags &= ~(MOV_TFHD_DEFAULT_SIZE | MOV_TFHD_DEFAULT_DURATION);



    avio_wb32(pb, 0); /* size placeholder */

    ffio_wfourcc(pb, "tfhd");

    avio_w8(pb, 0); /* version */

    avio_wb24(pb, flags);



    avio_wb32(pb, track->track_id); /* track-id */

    if (flags & MOV_TFHD_BASE_DATA_OFFSET)

        avio_wb64(pb, moof_offset);

    if (flags & MOV_TFHD_DEFAULT_DURATION) {

        track->default_duration = get_cluster_duration(track, 0);

        avio_wb32(pb, track->default_duration);

    }

    if (flags & MOV_TFHD_DEFAULT_SIZE) {

        track->default_size = track->entry ? track->cluster[0].size : 1;

        avio_wb32(pb, track->default_size);

    } else

        track->default_size = -1;



    if (flags & MOV_TFHD_DEFAULT_FLAGS) {

        track->default_sample_flags =

            track->enc->codec_type == AVMEDIA_TYPE_VIDEO ?

            (MOV_FRAG_SAMPLE_FLAG_DEPENDS_YES | MOV_FRAG_SAMPLE_FLAG_IS_NON_SYNC) :

            MOV_FRAG_SAMPLE_FLAG_DEPENDS_NO;

        avio_wb32(pb, track->default_sample_flags);

    }



    return update_size(pb, pos);

}

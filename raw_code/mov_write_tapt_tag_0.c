static int mov_write_tapt_tag(AVIOContext *pb, MOVTrack *track)

{

    int32_t width = av_rescale(track->enc->sample_aspect_ratio.num, track->enc->width,

                               track->enc->sample_aspect_ratio.den);



    int64_t pos = avio_tell(pb);



    avio_wb32(pb, 0); /* size */

    ffio_wfourcc(pb, "tapt");



    avio_wb32(pb, 20);

    ffio_wfourcc(pb, "clef");

    avio_wb32(pb, 0);

    avio_wb32(pb, width << 16);

    avio_wb32(pb, track->enc->height << 16);



    avio_wb32(pb, 20);

    ffio_wfourcc(pb, "enof");

    avio_wb32(pb, 0);

    avio_wb32(pb, track->enc->width << 16);

    avio_wb32(pb, track->enc->height << 16);



    return updateSize(pb, pos);

};

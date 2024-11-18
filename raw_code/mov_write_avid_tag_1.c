static int mov_write_avid_tag(AVIOContext *pb, MOVTrack *track)

{

    int i;

    avio_wb32(pb, 24); /* size */

    ffio_wfourcc(pb, "ACLR");

    ffio_wfourcc(pb, "ACLR");

    ffio_wfourcc(pb, "0001");

    if (track->enc->color_range == AVCOL_RANGE_MPEG || /* Legal range (16-235) */

        track->enc->color_range == AVCOL_RANGE_UNSPECIFIED) {

        avio_wb32(pb, 1); /* Corresponds to 709 in official encoder */

    } else { /* Full range (0-255) */

        avio_wb32(pb, 2); /* Corresponds to RGB in official encoder */

    }

    avio_wb32(pb, 0); /* unknown */



    avio_wb32(pb, 24); /* size */

    ffio_wfourcc(pb, "APRG");

    ffio_wfourcc(pb, "APRG");

    ffio_wfourcc(pb, "0001");

    avio_wb32(pb, 1); /* unknown */

    avio_wb32(pb, 0); /* unknown */



    avio_wb32(pb, 120); /* size */

    ffio_wfourcc(pb, "ARES");

    ffio_wfourcc(pb, "ARES");

    ffio_wfourcc(pb, "0001");

    avio_wb32(pb, AV_RB32(track->vos_data + 0x28)); /* dnxhd cid, some id ? */

    avio_wb32(pb, track->enc->width);

    /* values below are based on samples created with quicktime and avid codecs */

    if (track->vos_data[5] & 2) { // interlaced

        avio_wb32(pb, track->enc->height / 2);

        avio_wb32(pb, 2); /* unknown */

        avio_wb32(pb, 0); /* unknown */

        avio_wb32(pb, 4); /* unknown */

    } else {

        avio_wb32(pb, track->enc->height);

        avio_wb32(pb, 1); /* unknown */

        avio_wb32(pb, 0); /* unknown */

        if (track->enc->height == 1080)

            avio_wb32(pb, 5); /* unknown */

        else

            avio_wb32(pb, 6); /* unknown */

    }

    /* padding */

    for (i = 0; i < 10; i++)

        avio_wb64(pb, 0);



    /* extra padding for stsd needed */

    avio_wb32(pb, 0);

    return 0;

}

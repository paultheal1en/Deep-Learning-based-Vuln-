static int mov_write_colr_tag(AVIOContext *pb, MOVTrack *track)

{

    // Ref (MOV): https://developer.apple.com/library/mac/technotes/tn2162/_index.html#//apple_ref/doc/uid/DTS40013070-CH1-TNTAG9

    // Ref (MP4): ISO/IEC 14496-12:2012



    if (track->enc->color_primaries == AVCOL_PRI_UNSPECIFIED &&

        track->enc->color_trc == AVCOL_TRC_UNSPECIFIED &&

        track->enc->colorspace == AVCOL_SPC_UNSPECIFIED) {

        if ((track->enc->width >= 1920 && track->enc->height >= 1080)

          || (track->enc->width == 1280 && track->enc->height == 720)) {

            av_log(NULL, AV_LOG_WARNING, "color primaries unspecified, assuming bt709\n");

            track->enc->color_primaries = AVCOL_PRI_BT709;

        } else if (track->enc->width == 720 && track->height == 576) {

            av_log(NULL, AV_LOG_WARNING, "color primaries unspecified, assuming bt470bg\n");

            track->enc->color_primaries = AVCOL_PRI_BT470BG;

        } else if (track->enc->width == 720 &&

                   (track->height == 486 || track->height == 480)) {

            av_log(NULL, AV_LOG_WARNING, "color primaries unspecified, assuming smpte170\n");

            track->enc->color_primaries = AVCOL_PRI_SMPTE170M;

        } else {

            av_log(NULL, AV_LOG_WARNING, "color primaries unspecified, unable to assume anything\n");

        }

        switch (track->enc->color_primaries) {

        case AVCOL_PRI_BT709:

            track->enc->color_trc = AVCOL_TRC_BT709;

            track->enc->colorspace = AVCOL_SPC_BT709;

            break;

        case AVCOL_PRI_SMPTE170M:

        case AVCOL_PRI_BT470BG:

            track->enc->color_trc = AVCOL_TRC_BT709;

            track->enc->colorspace = AVCOL_SPC_SMPTE170M;

            break;

        }

    }



    /* We should only ever be called by MOV or MP4. */

    av_assert0(track->mode == MODE_MOV || track->mode == MODE_MP4);



    avio_wb32(pb, 18 + (track->mode == MODE_MP4));

    ffio_wfourcc(pb, "colr");

    if (track->mode == MODE_MP4)

        ffio_wfourcc(pb, "nclx");

    else

        ffio_wfourcc(pb, "nclc");

    switch (track->enc->color_primaries) {

    case AVCOL_PRI_BT709:     avio_wb16(pb, 1); break;

    case AVCOL_PRI_SMPTE170M:

    case AVCOL_PRI_SMPTE240M: avio_wb16(pb, 6); break;

    case AVCOL_PRI_BT470BG:   avio_wb16(pb, 5); break;

    default:                  avio_wb16(pb, 2);

    }

    switch (track->enc->color_trc) {

    case AVCOL_TRC_BT709:     avio_wb16(pb, 1); break;

    case AVCOL_TRC_SMPTE170M: avio_wb16(pb, 1); break; // remapped

    case AVCOL_TRC_SMPTE240M: avio_wb16(pb, 7); break;

    default:                  avio_wb16(pb, 2);

    }

    switch (track->enc->colorspace) {

    case AVCOL_TRC_BT709:     avio_wb16(pb, 1); break;

    case AVCOL_SPC_BT470BG:

    case AVCOL_PRI_SMPTE170M: avio_wb16(pb, 6); break;

    case AVCOL_PRI_SMPTE240M: avio_wb16(pb, 7); break;

    default:                  avio_wb16(pb, 2);

    }



    if (track->mode == MODE_MP4) {

        int full_range = track->enc->color_range == AVCOL_RANGE_JPEG;

        avio_w8(pb, full_range << 7);

        return 19;

    } else {

        return 18;

    }

}

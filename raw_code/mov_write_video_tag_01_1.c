static int mov_write_video_tag(AVIOContext *pb, MOVMuxContext *mov, MOVTrack *track)

{

    int64_t pos = avio_tell(pb);

    char compressor_name[32] = { 0 };



    avio_wb32(pb, 0); /* size */

    avio_wl32(pb, track->tag); // store it byteswapped

    avio_wb32(pb, 0); /* Reserved */

    avio_wb16(pb, 0); /* Reserved */

    avio_wb16(pb, 1); /* Data-reference index */



    avio_wb16(pb, 0); /* Codec stream version */

    avio_wb16(pb, 0); /* Codec stream revision (=0) */

    if (track->mode == MODE_MOV) {

        ffio_wfourcc(pb, "FFMP"); /* Vendor */

        if (track->enc->codec_id == AV_CODEC_ID_RAWVIDEO) {

            avio_wb32(pb, 0); /* Temporal Quality */

            avio_wb32(pb, 0x400); /* Spatial Quality = lossless*/

        } else {

            avio_wb32(pb, 0x200); /* Temporal Quality = normal */

            avio_wb32(pb, 0x200); /* Spatial Quality = normal */

        }

    } else {

        avio_wb32(pb, 0); /* Reserved */

        avio_wb32(pb, 0); /* Reserved */

        avio_wb32(pb, 0); /* Reserved */

    }

    avio_wb16(pb, track->enc->width); /* Video width */

    avio_wb16(pb, track->height); /* Video height */

    avio_wb32(pb, 0x00480000); /* Horizontal resolution 72dpi */

    avio_wb32(pb, 0x00480000); /* Vertical resolution 72dpi */

    avio_wb32(pb, 0); /* Data size (= 0) */

    avio_wb16(pb, 1); /* Frame count (= 1) */



    /* FIXME not sure, ISO 14496-1 draft where it shall be set to 0 */

    find_compressor(compressor_name, 32, track);

    avio_w8(pb, strlen(compressor_name));

    avio_write(pb, compressor_name, 31);



    if (track->mode == MODE_MOV && track->enc->bits_per_coded_sample)

        avio_wb16(pb, track->enc->bits_per_coded_sample);

    else

        avio_wb16(pb, 0x18); /* Reserved */

    avio_wb16(pb, 0xffff); /* Reserved */

    if (track->tag == MKTAG('m','p','4','v'))

        mov_write_esds_tag(pb, track);

    else if (track->enc->codec_id == AV_CODEC_ID_H263)

        mov_write_d263_tag(pb);

    else if (track->enc->codec_id == AV_CODEC_ID_AVUI ||

            track->enc->codec_id == AV_CODEC_ID_SVQ3) {

        mov_write_extradata_tag(pb, track);

        avio_wb32(pb, 0);

    } else if (track->enc->codec_id == AV_CODEC_ID_DNXHD)

        mov_write_avid_tag(pb, track);

    else if (track->enc->codec_id == AV_CODEC_ID_HEVC)

        mov_write_hvcc_tag(pb, track);

    else if (track->enc->codec_id == AV_CODEC_ID_H264 && !TAG_IS_AVCI(track->tag)) {

        mov_write_avcc_tag(pb, track);

        if (track->mode == MODE_IPOD)

            mov_write_uuid_tag_ipod(pb);

    } else if (track->enc->codec_id == AV_CODEC_ID_VC1 && track->vos_len > 0)

        mov_write_dvc1_tag(pb, track);

    else if (track->enc->codec_id == AV_CODEC_ID_VP6F ||

             track->enc->codec_id == AV_CODEC_ID_VP6A) {

        /* Don't write any potential extradata here - the cropping

         * is signalled via the normal width/height fields. */

    } else if (track->enc->codec_id == AV_CODEC_ID_R10K) {

        if (track->enc->codec_tag == MKTAG('R','1','0','k'))

            mov_write_dpxe_tag(pb, track);

    } else if (track->vos_len > 0)

        mov_write_glbl_tag(pb, track);



    if (track->enc->codec_id != AV_CODEC_ID_H264 &&

        track->enc->codec_id != AV_CODEC_ID_MPEG4 &&

        track->enc->codec_id != AV_CODEC_ID_DNXHD)

        if (track->enc->field_order != AV_FIELD_UNKNOWN)

            mov_write_fiel_tag(pb, track);



    if (mov->flags & FF_MOV_FLAG_WRITE_COLR)

        mov_write_colr_tag(pb, track);



    if (track->enc->sample_aspect_ratio.den && track->enc->sample_aspect_ratio.num &&

        track->enc->sample_aspect_ratio.den != track->enc->sample_aspect_ratio.num) {

        mov_write_pasp_tag(pb, track);

    }



    return update_size(pb, pos);

}

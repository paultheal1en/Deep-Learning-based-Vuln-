void ff_mpeg1_encode_picture_header(MpegEncContext *s, int picture_number)

{

    AVFrameSideData *side_data;

    mpeg1_encode_sequence_header(s);



    /* mpeg1 picture header */

    put_header(s, PICTURE_START_CODE);

    /* temporal reference */



    // RAL: s->picture_number instead of s->fake_picture_number

    put_bits(&s->pb, 10,

             (s->picture_number - s->gop_picture_number) & 0x3ff);

    put_bits(&s->pb, 3, s->pict_type);



    s->vbv_delay_ptr = s->pb.buf + put_bits_count(&s->pb) / 8;

    put_bits(&s->pb, 16, 0xFFFF);               /* vbv_delay */



    // RAL: Forward f_code also needed for B-frames

    if (s->pict_type == AV_PICTURE_TYPE_P ||

        s->pict_type == AV_PICTURE_TYPE_B) {

        put_bits(&s->pb, 1, 0);                 /* half pel coordinates */

        if (s->codec_id == AV_CODEC_ID_MPEG1VIDEO)

            put_bits(&s->pb, 3, s->f_code);     /* forward_f_code */

        else

            put_bits(&s->pb, 3, 7);             /* forward_f_code */

    }



    // RAL: Backward f_code necessary for B-frames

    if (s->pict_type == AV_PICTURE_TYPE_B) {

        put_bits(&s->pb, 1, 0);                 /* half pel coordinates */

        if (s->codec_id == AV_CODEC_ID_MPEG1VIDEO)

            put_bits(&s->pb, 3, s->b_code);     /* backward_f_code */

        else

            put_bits(&s->pb, 3, 7);             /* backward_f_code */

    }



    put_bits(&s->pb, 1, 0);                     /* extra bit picture */



    s->frame_pred_frame_dct = 1;

    if (s->codec_id == AV_CODEC_ID_MPEG2VIDEO) {

        put_header(s, EXT_START_CODE);

        put_bits(&s->pb, 4, 8);                 /* pic ext */

        if (s->pict_type == AV_PICTURE_TYPE_P ||

            s->pict_type == AV_PICTURE_TYPE_B) {

            put_bits(&s->pb, 4, s->f_code);

            put_bits(&s->pb, 4, s->f_code);

        } else {

            put_bits(&s->pb, 8, 255);

        }

        if (s->pict_type == AV_PICTURE_TYPE_B) {

            put_bits(&s->pb, 4, s->b_code);

            put_bits(&s->pb, 4, s->b_code);

        } else {

            put_bits(&s->pb, 8, 255);

        }

        put_bits(&s->pb, 2, s->intra_dc_precision);



        assert(s->picture_structure == PICT_FRAME);

        put_bits(&s->pb, 2, s->picture_structure);

        if (s->progressive_sequence)

            put_bits(&s->pb, 1, 0);             /* no repeat */

        else

            put_bits(&s->pb, 1, s->current_picture_ptr->f.top_field_first);

        /* XXX: optimize the generation of this flag with entropy measures */

        s->frame_pred_frame_dct = s->progressive_sequence;



        put_bits(&s->pb, 1, s->frame_pred_frame_dct);

        put_bits(&s->pb, 1, s->concealment_motion_vectors);

        put_bits(&s->pb, 1, s->q_scale_type);

        put_bits(&s->pb, 1, s->intra_vlc_format);

        put_bits(&s->pb, 1, s->alternate_scan);

        put_bits(&s->pb, 1, s->repeat_first_field);

        s->progressive_frame = s->progressive_sequence;

        /* chroma_420_type */

        put_bits(&s->pb, 1, s->chroma_format ==

                            CHROMA_420 ? s->progressive_frame : 0);

        put_bits(&s->pb, 1, s->progressive_frame);

        put_bits(&s->pb, 1, 0);                 /* composite_display_flag */

    }

    if (s->scan_offset) {

        int i;



        put_header(s, USER_START_CODE);

        for (i = 0; i < sizeof(svcd_scan_offset_placeholder); i++)

            put_bits(&s->pb, 8, svcd_scan_offset_placeholder[i]);

    }

    side_data = av_frame_get_side_data(&s->current_picture_ptr->f,

                                       AV_FRAME_DATA_STEREO3D);

    if (side_data) {

        AVStereo3D *stereo = (AVStereo3D *)side_data->data;

        uint8_t fpa_type;



        switch (stereo->type) {

        case AV_STEREO3D_SIDEBYSIDE:

            fpa_type = 0x03;

            break;

        case AV_STEREO3D_TOPBOTTOM:

            fpa_type = 0x04;

            break;

        case AV_STEREO3D_2D:

            fpa_type = 0x08;

            break;

        case AV_STEREO3D_SIDEBYSIDE_QUINCUNX:

            fpa_type = 0x23;

            break;

        default:

            fpa_type = 0;

            break;

        }



        if (fpa_type != 0) {

            put_header(s, USER_START_CODE);

            put_bits(&s->pb, 8, 'J');   // S3D_video_format_signaling_identifier

            put_bits(&s->pb, 8, 'P');

            put_bits(&s->pb, 8, '3');

            put_bits(&s->pb, 8, 'D');

            put_bits(&s->pb, 8, 0x03);  // S3D_video_format_length



            put_bits(&s->pb, 1, 1);     // reserved_bit

            put_bits(&s->pb, 7, fpa_type); // S3D_video_format_type

            put_bits(&s->pb, 8, 0x04);  // reserved_data[0]

            put_bits(&s->pb, 8, 0xFF);  // reserved_data[1]

        }

    }



    s->mb_y = 0;

    ff_mpeg1_encode_slice_header(s);

}

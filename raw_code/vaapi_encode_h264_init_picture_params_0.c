static int vaapi_encode_h264_init_picture_params(AVCodecContext *avctx,

                                                 VAAPIEncodePicture *pic)

{

    VAAPIEncodeContext                *ctx = avctx->priv_data;

    VAEncSequenceParameterBufferH264 *vseq = ctx->codec_sequence_params;

    VAEncPictureParameterBufferH264  *vpic = pic->codec_picture_params;

    VAAPIEncodeH264Context           *priv = ctx->priv_data;

    int i;



    if (pic->type == PICTURE_TYPE_IDR) {

        av_assert0(pic->display_order == pic->encode_order);

        vpic->frame_num = 0;

        priv->next_frame_num = 1;

        priv->cpb_delay = 0;

    } else {

        vpic->frame_num = priv->next_frame_num;

        if (pic->type != PICTURE_TYPE_B) {

            // nal_ref_idc != 0

            ++priv->next_frame_num;

        }

        ++priv->cpb_delay;

    }

    priv->dpb_delay = pic->display_order - pic->encode_order + 1;



    vpic->frame_num = vpic->frame_num &

        ((1 << (4 + vseq->seq_fields.bits.log2_max_frame_num_minus4)) - 1);



    vpic->CurrPic.picture_id          = pic->recon_surface;

    vpic->CurrPic.frame_idx           = vpic->frame_num;

    vpic->CurrPic.flags               = 0;

    vpic->CurrPic.TopFieldOrderCnt    = pic->display_order;

    vpic->CurrPic.BottomFieldOrderCnt = pic->display_order;



    for (i = 0; i < pic->nb_refs; i++) {

        VAAPIEncodePicture *ref = pic->refs[i];

        av_assert0(ref && ref->encode_order < pic->encode_order);

        vpic->ReferenceFrames[i].picture_id = ref->recon_surface;

        vpic->ReferenceFrames[i].frame_idx  = ref->encode_order;

        vpic->ReferenceFrames[i].flags = VA_PICTURE_H264_SHORT_TERM_REFERENCE;

        vpic->ReferenceFrames[i].TopFieldOrderCnt    = ref->display_order;

        vpic->ReferenceFrames[i].BottomFieldOrderCnt = ref->display_order;

    }

    for (; i < FF_ARRAY_ELEMS(vpic->ReferenceFrames); i++) {

        vpic->ReferenceFrames[i].picture_id = VA_INVALID_ID;

        vpic->ReferenceFrames[i].flags = VA_PICTURE_H264_INVALID;

    }



    vpic->coded_buf = pic->output_buffer;



    vpic->pic_fields.bits.idr_pic_flag = (pic->type == PICTURE_TYPE_IDR);

    vpic->pic_fields.bits.reference_pic_flag = (pic->type != PICTURE_TYPE_B);



    pic->nb_slices = 1;



    return 0;

}

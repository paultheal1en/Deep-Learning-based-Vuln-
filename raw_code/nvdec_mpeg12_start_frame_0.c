static int nvdec_mpeg12_start_frame(AVCodecContext *avctx, const uint8_t *buffer, uint32_t size)

{

    MpegEncContext *s = avctx->priv_data;



    NVDECContext      *ctx = avctx->internal->hwaccel_priv_data;

    CUVIDPICPARAMS     *pp = &ctx->pic_params;

    CUVIDMPEG2PICPARAMS *ppc = &pp->CodecSpecific.mpeg2;

    FrameDecodeData *fdd;

    NVDECFrame *cf;

    AVFrame *cur_frame = s->current_picture.f;



    int ret, i;



    ret = ff_nvdec_start_frame(avctx, cur_frame);

    if (ret < 0)

        return ret;



    fdd = (FrameDecodeData*)cur_frame->private_ref->data;

    cf  = (NVDECFrame*)fdd->hwaccel_priv;



    *pp = (CUVIDPICPARAMS) {

        .PicWidthInMbs     = (cur_frame->width  + 15) / 16,

        .FrameHeightInMbs  = (cur_frame->height + 15) / 16,

        .CurrPicIdx        = cf->idx,



        .intra_pic_flag    = s->pict_type == AV_PICTURE_TYPE_I,

        .ref_pic_flag      = s->pict_type == AV_PICTURE_TYPE_I ||

                             s->pict_type == AV_PICTURE_TYPE_P,



        .CodecSpecific.mpeg2 = {

            .ForwardRefIdx     = get_ref_idx(s->last_picture.f),

            .BackwardRefIdx    = get_ref_idx(s->next_picture.f),



            .picture_coding_type        = s->pict_type,

            .full_pel_forward_vector    = s->full_pel[0],

            .full_pel_backward_vector   = s->full_pel[1],

            .f_code                     = { { s->mpeg_f_code[0][0],

                                              s->mpeg_f_code[0][1] },

                                            { s->mpeg_f_code[1][0],

                                              s->mpeg_f_code[1][1] } },

            .intra_dc_precision         = s->intra_dc_precision,

            .frame_pred_frame_dct       = s->frame_pred_frame_dct,

            .concealment_motion_vectors = s->concealment_motion_vectors,

            .q_scale_type               = s->q_scale_type,

            .intra_vlc_format           = s->intra_vlc_format,

            .alternate_scan             = s->alternate_scan,

            .top_field_first            = s->top_field_first,

        }

    };



    for (i = 0; i < 64; ++i) {

        ppc->QuantMatrixIntra[i] = s->intra_matrix[i];

        ppc->QuantMatrixInter[i] = s->inter_matrix[i];

    }



    return 0;

}

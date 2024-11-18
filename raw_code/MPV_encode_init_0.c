int MPV_encode_init(AVCodecContext *avctx)

{

    MpegEncContext *s = avctx->priv_data;

    int i;



    avctx->pix_fmt = PIX_FMT_YUV420P;



    s->bit_rate = avctx->bit_rate;

    s->bit_rate_tolerance = avctx->bit_rate_tolerance;

    s->frame_rate = avctx->frame_rate;

    s->width = avctx->width;

    s->height = avctx->height;

    s->gop_size = avctx->gop_size;

    s->rtp_mode = avctx->rtp_mode;

    s->rtp_payload_size = avctx->rtp_payload_size;

    if (avctx->rtp_callback)

        s->rtp_callback = avctx->rtp_callback;

    s->qmin= avctx->qmin;

    s->qmax= avctx->qmax;

    s->max_qdiff= avctx->max_qdiff;

    s->qcompress= avctx->qcompress;

    s->qblur= avctx->qblur;

    s->b_quant_factor= avctx->b_quant_factor;

    s->avctx = avctx;

    s->aspect_ratio_info= avctx->aspect_ratio_info;

    s->flags= avctx->flags;

    s->max_b_frames= avctx->max_b_frames;

    s->rc_strategy= avctx->rc_strategy;

    s->b_frame_strategy= avctx->b_frame_strategy;

    s->codec_id= avctx->codec->id;



    if (s->gop_size <= 1) {

        s->intra_only = 1;

        s->gop_size = 12;

    } else {

        s->intra_only = 0;

    }

    

    /* ME algorithm */

    if (avctx->me_method == 0)

        /* For compatibility */

        s->me_method = motion_estimation_method;

    else

        s->me_method = avctx->me_method;

        

    /* Fixed QSCALE */

    s->fixed_qscale = (avctx->flags & CODEC_FLAG_QSCALE);

    

    switch(avctx->codec->id) {

    case CODEC_ID_MPEG1VIDEO:

        s->out_format = FMT_MPEG1;

        avctx->delay=0; //FIXME not sure, should check the spec

        break;

    case CODEC_ID_MJPEG:

        s->out_format = FMT_MJPEG;

        s->intra_only = 1; /* force intra only for jpeg */

        s->mjpeg_write_tables = 1; /* write all tables */

	s->mjpeg_data_only_frames = 0; /* write all the needed headers */

        s->mjpeg_vsample[0] = 2; /* set up default sampling factors */

        s->mjpeg_vsample[1] = 1; /* the only currently supported values */

        s->mjpeg_vsample[2] = 1; 

        s->mjpeg_hsample[0] = 2;

        s->mjpeg_hsample[1] = 1; 

        s->mjpeg_hsample[2] = 1; 

        if (mjpeg_init(s) < 0)

            return -1;

        avctx->delay=0;

        break;

    case CODEC_ID_H263:

        if (h263_get_picture_format(s->width, s->height) == 7) {

            printf("Input picture size isn't suitable for h263 codec! try h263+\n");

            return -1;

        }

        s->out_format = FMT_H263;

        avctx->delay=0;

        break;

    case CODEC_ID_H263P:

        s->out_format = FMT_H263;

        s->rtp_mode = 1;

        s->rtp_payload_size = 1200; 

        s->h263_plus = 1;

        s->unrestricted_mv = 1;

        s->h263_aic = 1;

        

        /* These are just to be sure */

        s->umvplus = 0;

        s->umvplus_dec = 0;

        avctx->delay=0;

        break;

    case CODEC_ID_RV10:

        s->out_format = FMT_H263;

        s->h263_rv10 = 1;

        avctx->delay=0;

        break;

    case CODEC_ID_MPEG4:

        s->out_format = FMT_H263;

        s->h263_pred = 1;

        s->unrestricted_mv = 1;

        s->has_b_frames= s->max_b_frames ? 1 : 0;

        s->low_delay=0;

        avctx->delay= s->low_delay ? 0 : (s->max_b_frames + 1); 

        break;

    case CODEC_ID_MSMPEG4V1:

        s->out_format = FMT_H263;

        s->h263_msmpeg4 = 1;

        s->h263_pred = 1;

        s->unrestricted_mv = 1;

        s->msmpeg4_version= 1;

        avctx->delay=0;

        break;

    case CODEC_ID_MSMPEG4V2:

        s->out_format = FMT_H263;

        s->h263_msmpeg4 = 1;

        s->h263_pred = 1;

        s->unrestricted_mv = 1;

        s->msmpeg4_version= 2;

        avctx->delay=0;

        break;

    case CODEC_ID_MSMPEG4V3:

        s->out_format = FMT_H263;

        s->h263_msmpeg4 = 1;

        s->h263_pred = 1;

        s->unrestricted_mv = 1;

        s->msmpeg4_version= 3;

        avctx->delay=0;

        break;

    default:

        return -1;

    }

    

    if((s->flags&CODEC_FLAG_4MV) && !(s->flags&CODEC_FLAG_HQ)){

        printf("4MV is currently only supported in HQ mode\n");

        return -1;

    }



    { /* set up some save defaults, some codecs might override them later */

        static int done=0;

        if(!done){

            int i;

            done=1;

            memset(default_mv_penalty, 0, sizeof(UINT16)*(MAX_FCODE+1)*(2*MAX_MV+1));

            memset(default_fcode_tab , 0, sizeof(UINT8)*(2*MAX_MV+1));



            for(i=-16; i<16; i++){

                default_fcode_tab[i + MAX_MV]= 1;

            }

        }

    }

    s->mv_penalty= default_mv_penalty;

    s->fcode_tab= default_fcode_tab;



    if (s->out_format == FMT_H263)

        h263_encode_init(s);

    else if (s->out_format == FMT_MPEG1)

        mpeg1_encode_init(s);



    /* dont use mv_penalty table for crap MV as it would be confused */

    if (s->me_method < ME_EPZS) s->mv_penalty = default_mv_penalty;



    s->encoding = 1;



    /* init */

    if (MPV_common_init(s) < 0)

        return -1;

    

    /* init default q matrix */

    for(i=0;i<64;i++) {

        if(s->out_format == FMT_H263)

            s->intra_matrix[i] = default_non_intra_matrix[i];

        else

            s->intra_matrix[i] = default_intra_matrix[i];



        s->inter_matrix[i] = default_non_intra_matrix[i];

    }



    /* precompute matrix */

    /* for mjpeg, we do include qscale in the matrix */

    if (s->out_format != FMT_MJPEG) {

        convert_matrix(s->q_intra_matrix, s->q_intra_matrix16, s->q_intra_matrix16_bias, 

                       s->intra_matrix, s->intra_quant_bias);

        convert_matrix(s->q_inter_matrix, s->q_inter_matrix16, s->q_inter_matrix16_bias, 

                       s->inter_matrix, s->inter_quant_bias);

    }



    if(ff_rate_control_init(s) < 0)

        return -1;



    s->picture_number = 0;

    s->picture_in_gop_number = 0;

    s->fake_picture_number = 0;

    /* motion detector init */

    s->f_code = 1;

    s->b_code = 1;



    return 0;

}

static void check_default_settings(AVCodecContext *avctx)

{

    X264Context *x4 = avctx->priv_data;



    int score = 0;

    score += x4->params.analyse.i_me_range == 0;

    score += x4->params.rc.i_qp_step == 3;

    score += x4->params.i_keyint_max == 12;

    score += x4->params.rc.i_qp_min == 2;

    score += x4->params.rc.i_qp_max == 31;

    score += x4->params.rc.f_qcompress == 0.5;

    score += fabs(x4->params.rc.f_ip_factor - 1.25) < 0.01;

    score += fabs(x4->params.rc.f_pb_factor - 1.25) < 0.01;

    score += x4->params.analyse.inter == 0 && x4->params.analyse.i_subpel_refine == 8;

    if (score >= 5) {

        av_log(avctx, AV_LOG_ERROR, "Default settings detected, using medium profile\n");

        x4->preset = av_strdup("medium");

        if (avctx->bit_rate == 200*1000)

            avctx->crf = 23;

    }

}

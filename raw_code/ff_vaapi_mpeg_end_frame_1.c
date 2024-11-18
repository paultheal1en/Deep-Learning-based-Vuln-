int ff_vaapi_mpeg_end_frame(AVCodecContext *avctx)

{

    struct vaapi_context * const vactx = avctx->hwaccel_context;

    MpegEncContext *s = avctx->priv_data;

    int ret;



    ret = ff_vaapi_commit_slices(vactx);

    if (ret < 0)

        goto finish;



    ret = ff_vaapi_render_picture(vactx,

                                  ff_vaapi_get_surface_id(&s->current_picture_ptr->f));

    if (ret < 0)

        goto finish;



    ff_mpeg_draw_horiz_band(s, 0, s->avctx->height);



finish:

    ff_vaapi_common_end_frame(avctx);

    return ret;

}

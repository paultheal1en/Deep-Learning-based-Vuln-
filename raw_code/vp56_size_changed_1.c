static int vp56_size_changed(VP56Context *s)
{
    AVCodecContext *avctx = s->avctx;
    int stride = s->frames[VP56_FRAME_CURRENT]->linesize[0];
    int i;
    s->plane_width[0]  = s->plane_width[3]  = avctx->coded_width;
    s->plane_width[1]  = s->plane_width[2]  = avctx->coded_width/2;
    s->plane_height[0] = s->plane_height[3] = avctx->coded_height;
    s->plane_height[1] = s->plane_height[2] = avctx->coded_height/2;
    for (i=0; i<4; i++)
        s->stride[i] = s->flip * s->frames[VP56_FRAME_CURRENT]->linesize[i];
    s->mb_width  = (avctx->coded_width +15) / 16;
    s->mb_height = (avctx->coded_height+15) / 16;
    if (s->mb_width > 1000 || s->mb_height > 1000) {
        ff_set_dimensions(avctx, 0, 0);
        av_log(avctx, AV_LOG_ERROR, "picture too big\n");
        return AVERROR_INVALIDDATA;
    }
    av_reallocp_array(&s->above_blocks, 4*s->mb_width+6,
                      sizeof(*s->above_blocks));
    av_reallocp_array(&s->macroblocks, s->mb_width*s->mb_height,
                      sizeof(*s->macroblocks));
    av_free(s->edge_emu_buffer_alloc);
    s->edge_emu_buffer_alloc = av_malloc(16*stride);
    s->edge_emu_buffer = s->edge_emu_buffer_alloc;
    if (!s->above_blocks || !s->macroblocks || !s->edge_emu_buffer_alloc)
        return AVERROR(ENOMEM);
    if (s->flip < 0)
        s->edge_emu_buffer += 15 * stride;
    if (s->alpha_context)
        return vp56_size_changed(s->alpha_context);
    return 0;
}
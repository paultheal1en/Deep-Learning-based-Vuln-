int ff_vaapi_render_picture(struct vaapi_context *vactx, VASurfaceID surface)
{
    VABufferID va_buffers[3];
    unsigned int n_va_buffers = 0;
    vaUnmapBuffer(vactx->display, vactx->pic_param_buf_id);
    va_buffers[n_va_buffers++] = vactx->pic_param_buf_id;
    if (vactx->iq_matrix_buf_id) {
        vaUnmapBuffer(vactx->display, vactx->iq_matrix_buf_id);
        va_buffers[n_va_buffers++] = vactx->iq_matrix_buf_id;
    }
    if (vactx->bitplane_buf_id) {
        vaUnmapBuffer(vactx->display, vactx->bitplane_buf_id);
        va_buffers[n_va_buffers++] = vactx->bitplane_buf_id;
    }
    if (vaBeginPicture(vactx->display, vactx->context_id,
                       surface) != VA_STATUS_SUCCESS)
        return -1;
    if (vaRenderPicture(vactx->display, vactx->context_id,
                        va_buffers, n_va_buffers) != VA_STATUS_SUCCESS)
        return -1;
    if (vaRenderPicture(vactx->display, vactx->context_id,
                        vactx->slice_buf_ids,
                        vactx->n_slice_buf_ids) != VA_STATUS_SUCCESS)
        return -1;
    if (vaEndPicture(vactx->display, vactx->context_id) != VA_STATUS_SUCCESS)
        return -1;
}
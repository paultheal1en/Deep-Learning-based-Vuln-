static void *alloc_buffer(FFVAContext *vactx, int type, unsigned int size, uint32_t *buf_id)

{

    void *data = NULL;



    *buf_id = 0;

    if (vaCreateBuffer(vactx->display, vactx->context_id,

                       type, size, 1, NULL, buf_id) == VA_STATUS_SUCCESS)

        vaMapBuffer(vactx->display, *buf_id, &data);



    return data;

}

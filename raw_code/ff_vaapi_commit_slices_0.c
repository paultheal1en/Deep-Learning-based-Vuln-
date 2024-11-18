int ff_vaapi_commit_slices(FFVAContext *vactx)

{

    VABufferID *slice_buf_ids;

    VABufferID slice_param_buf_id, slice_data_buf_id;



    if (vactx->slice_count == 0)

        return 0;



    slice_buf_ids =

        av_fast_realloc(vactx->slice_buf_ids,

                        &vactx->slice_buf_ids_alloc,

                        (vactx->n_slice_buf_ids + 2) * sizeof(slice_buf_ids[0]));

    if (!slice_buf_ids)

        return -1;

    vactx->slice_buf_ids = slice_buf_ids;



    slice_param_buf_id = 0;

    if (vaCreateBuffer(vactx->display, vactx->context_id,

                       VASliceParameterBufferType,

                       vactx->slice_param_size,

                       vactx->slice_count, vactx->slice_params,

                       &slice_param_buf_id) != VA_STATUS_SUCCESS)

        return -1;

    vactx->slice_count = 0;



    slice_data_buf_id = 0;

    if (vaCreateBuffer(vactx->display, vactx->context_id,

                       VASliceDataBufferType,

                       vactx->slice_data_size,

                       1, (void *)vactx->slice_data,

                       &slice_data_buf_id) != VA_STATUS_SUCCESS)

        return -1;

    vactx->slice_data = NULL;

    vactx->slice_data_size = 0;



    slice_buf_ids[vactx->n_slice_buf_ids++] = slice_param_buf_id;

    slice_buf_ids[vactx->n_slice_buf_ids++] = slice_data_buf_id;

    return 0;

}

static void nvdimm_init_fit_buffer(NvdimmFitBuffer *fit_buf)

{

    qemu_mutex_init(&fit_buf->lock);

    fit_buf->fit = g_array_new(false, true /* clear */, 1);

}

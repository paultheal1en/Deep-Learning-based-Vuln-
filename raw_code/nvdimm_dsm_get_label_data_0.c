static void nvdimm_dsm_get_label_data(NVDIMMDevice *nvdimm, NvdimmDsmIn *in,

                                      hwaddr dsm_mem_addr)

{

    NVDIMMClass *nvc = NVDIMM_GET_CLASS(nvdimm);

    NvdimmFuncGetLabelDataIn *get_label_data;

    NvdimmFuncGetLabelDataOut *get_label_data_out;

    uint32_t status;

    int size;



    get_label_data = (NvdimmFuncGetLabelDataIn *)in->arg3;

    le32_to_cpus(&get_label_data->offset);

    le32_to_cpus(&get_label_data->length);



    nvdimm_debug("Read Label Data: offset %#x length %#x.\n",

                 get_label_data->offset, get_label_data->length);



    status = nvdimm_rw_label_data_check(nvdimm, get_label_data->offset,

                                        get_label_data->length);

    if (status != 0 /* Success */) {

        nvdimm_dsm_no_payload(status, dsm_mem_addr);

        return;

    }



    size = sizeof(*get_label_data_out) + get_label_data->length;

    assert(size <= 4096);

    get_label_data_out = g_malloc(size);



    get_label_data_out->len = cpu_to_le32(size);

    get_label_data_out->func_ret_status = cpu_to_le32(0 /* Success */);

    nvc->read_label_data(nvdimm, get_label_data_out->out_buf,

                         get_label_data->length, get_label_data->offset);



    cpu_physical_memory_write(dsm_mem_addr, get_label_data_out, size);

    g_free(get_label_data_out);

}

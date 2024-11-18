static void nvdimm_dsm_label_size(NVDIMMDevice *nvdimm, hwaddr dsm_mem_addr)

{

    NvdimmFuncGetLabelSizeOut label_size_out = {

        .len = cpu_to_le32(sizeof(label_size_out)),

    };

    uint32_t label_size, mxfer;



    label_size = nvdimm->label_size;

    mxfer = nvdimm_get_max_xfer_label_size();



    nvdimm_debug("label_size %#x, max_xfer %#x.\n", label_size, mxfer);



    label_size_out.func_ret_status = cpu_to_le32(0 /* Success */);

    label_size_out.label_size = cpu_to_le32(label_size);

    label_size_out.max_xfer = cpu_to_le32(mxfer);



    cpu_physical_memory_write(dsm_mem_addr, &label_size_out,

                              sizeof(label_size_out));

}

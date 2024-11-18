static uint32_t nvdimm_rw_label_data_check(NVDIMMDevice *nvdimm,

                                           uint32_t offset, uint32_t length)

{

    uint32_t ret = 3 /* Invalid Input Parameters */;



    if (offset + length < offset) {

        nvdimm_debug("offset %#x + length %#x is overflow.\n", offset,

                     length);

        return ret;

    }



    if (nvdimm->label_size < offset + length) {

        nvdimm_debug("position %#x is beyond label data (len = %" PRIx64 ").\n",

                     offset + length, nvdimm->label_size);

        return ret;

    }



    if (length > nvdimm_get_max_xfer_label_size()) {

        nvdimm_debug("length (%#x) is larger than max_xfer (%#x).\n",

                     length, nvdimm_get_max_xfer_label_size());

        return ret;

    }



    return 0 /* Success */;

}

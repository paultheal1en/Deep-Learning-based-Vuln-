static uint64_t fw_cfg_data_mem_read(void *opaque, hwaddr addr,

                                     unsigned size)

{

    FWCfgState *s = opaque;

    uint8_t buf[8];

    unsigned i;



    for (i = 0; i < size; ++i) {

        buf[i] = fw_cfg_read(s);

    }

    switch (size) {

    case 1:

        return buf[0];

    case 2:

        return lduw_he_p(buf);

    case 4:

        return (uint32_t)ldl_he_p(buf);

    case 8:

        return ldq_he_p(buf);

    }

    abort();

}

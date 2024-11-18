static target_ulong h_get_term_char(CPUState *env, sPAPREnvironment *spapr,

                                    target_ulong opcode, target_ulong *args)

{

    target_ulong reg = args[0];

    target_ulong *len = args + 0;

    target_ulong *char0_7 = args + 1;

    target_ulong *char8_15 = args + 2;

    VIOsPAPRDevice *sdev = spapr_vio_find_by_reg(spapr->vio_bus, reg);

    uint8_t buf[16];



    if (!sdev) {

        return H_PARAMETER;

    }



    *len = vty_getchars(sdev, buf, sizeof(buf));

    if (*len < 16) {

        memset(buf + *len, 0, 16 - *len);

    }



    *char0_7 = be64_to_cpu(*((uint64_t *)buf));

    *char8_15 = be64_to_cpu(*((uint64_t *)buf + 1));



    return H_SUCCESS;

}

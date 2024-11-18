static void gen_flt3_ldst (DisasContext *ctx, uint32_t opc, int fd,

                           int fs, int base, int index)

{

    const char *opn = "extended float load/store";

    int store = 0;



    /* All of those work only on 64bit FPUs. */

    gen_op_cp1_64bitmode();

    if (base == 0) {

        if (index == 0)

            gen_op_reset_T0();

        else

            GEN_LOAD_REG_TN(T0, index);

    } else if (index == 0) {

        GEN_LOAD_REG_TN(T0, base);

    } else {

        GEN_LOAD_REG_TN(T0, base);

        GEN_LOAD_REG_TN(T1, index);

        gen_op_addr_add();

    }

    /* Don't do NOP if destination is zero: we must perform the actual

     * memory access

     */

    switch (opc) {

    case OPC_LWXC1:

        op_ldst(lwc1);

        GEN_STORE_FTN_FREG(fd, WT0);

        opn = "lwxc1";

        break;

    case OPC_LDXC1:

        op_ldst(ldc1);

        GEN_STORE_FTN_FREG(fd, DT0);

        opn = "ldxc1";

        break;

    case OPC_LUXC1:

        op_ldst(luxc1);

        GEN_STORE_FTN_FREG(fd, DT0);

        opn = "luxc1";

        break;

    case OPC_SWXC1:

        GEN_LOAD_FREG_FTN(WT0, fs);

        op_ldst(swc1);

        opn = "swxc1";

        store = 1;

        break;

    case OPC_SDXC1:

        GEN_LOAD_FREG_FTN(DT0, fs);

        op_ldst(sdc1);

        opn = "sdxc1";

        store = 1;

        break;

    case OPC_SUXC1:

        GEN_LOAD_FREG_FTN(DT0, fs);

        op_ldst(suxc1);

        opn = "suxc1";

        store = 1;

        break;

    default:

        MIPS_INVAL(opn);

        generate_exception(ctx, EXCP_RI);

        return;

    }

    MIPS_DEBUG("%s %s, %s(%s)", opn, fregnames[store ? fs : fd],

               regnames[index], regnames[base]);

}

static void gen_op_iwmmxt_setpsr_nz(void)

{

    TCGv tmp = new_tmp();

    gen_helper_iwmmxt_setpsr_nz(tmp, cpu_M0);

    store_cpu_field(tmp, iwmmxt.cregs[ARM_IWMMXT_wCASF]);

}

static target_ulong h_rtas(PowerPCCPU *cpu, sPAPREnvironment *spapr,

                           target_ulong opcode, target_ulong *args)

{

    target_ulong rtas_r3 = args[0];

    uint32_t token = ldl_be_phys(rtas_r3);

    uint32_t nargs = ldl_be_phys(rtas_r3 + 4);

    uint32_t nret = ldl_be_phys(rtas_r3 + 8);



    return spapr_rtas_call(spapr, token, nargs, rtas_r3 + 12,

                           nret, rtas_r3 + 12 + 4*nargs);

}

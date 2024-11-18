static target_ulong h_remove(PowerPCCPU *cpu, sPAPRMachineState *spapr,

                             target_ulong opcode, target_ulong *args)

{

    CPUPPCState *env = &cpu->env;

    target_ulong flags = args[0];

    target_ulong pte_index = args[1];

    target_ulong avpn = args[2];

    RemoveResult ret;



    ret = remove_hpte(cpu, pte_index, avpn, flags,

                      &args[0], &args[1]);



    switch (ret) {

    case REMOVE_SUCCESS:

        check_tlb_flush(env);

        return H_SUCCESS;



    case REMOVE_NOT_FOUND:

        return H_NOT_FOUND;



    case REMOVE_PARM:

        return H_PARAMETER;



    case REMOVE_HW:

        return H_HARDWARE;

    }



    g_assert_not_reached();

}

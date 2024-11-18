static target_ulong h_register_process_table(PowerPCCPU *cpu,

                                             sPAPRMachineState *spapr,

                                             target_ulong opcode,

                                             target_ulong *args)

{

    qemu_log_mask(LOG_UNIMP, "Unimplemented SPAPR hcall 0x"TARGET_FMT_lx"%s\n",

                  opcode, " (H_REGISTER_PROC_TBL)");

    return H_FUNCTION;

}

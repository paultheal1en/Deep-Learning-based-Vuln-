static void pc_i440fx_2_4_machine_options(MachineClass *m)

{

    PCMachineClass *pcmc = PC_MACHINE_CLASS(m);

    pc_i440fx_2_5_machine_options(m);

    m->alias = NULL;

    m->is_default = 0;

    pcmc->broken_reserved_end = true;

    pcmc->inter_dimm_gap = false;

    SET_MACHINE_COMPAT(m, PC_COMPAT_2_4);

}

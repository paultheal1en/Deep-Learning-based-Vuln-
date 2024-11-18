static void spapr_machine_2_5_class_options(MachineClass *mc)

{

    sPAPRMachineClass *smc = SPAPR_MACHINE_CLASS(mc);



    mc->alias = "pseries";

    mc->is_default = 1;

    smc->dr_lmb_enabled = true;

}

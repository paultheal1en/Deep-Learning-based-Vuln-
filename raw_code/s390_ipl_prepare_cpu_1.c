void s390_ipl_prepare_cpu(S390CPU *cpu)

{

    S390IPLState *ipl = get_ipl_device();




    cpu->env.psw.addr = ipl->start_addr;

    cpu->env.psw.mask = IPL_PSW_MASK;



    if (!ipl->kernel || ipl->iplb_valid) {

        cpu->env.psw.addr = ipl->bios_start_addr;

        if (!ipl->iplb_valid) {

            ipl->iplb_valid = s390_gen_initial_iplb(ipl);

        }

    }

    if (ipl->netboot) {

        if (load_netboot_image(&err) < 0) {

            error_report_err(err);

            vm_stop(RUN_STATE_INTERNAL_ERROR);

        }

        ipl->iplb.ccw.netboot_start_addr = ipl->start_addr;

    }

}
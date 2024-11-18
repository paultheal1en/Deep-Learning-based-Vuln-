static target_ulong h_client_architecture_support(PowerPCCPU *cpu_,

                                                  sPAPRMachineState *spapr,

                                                  target_ulong opcode,

                                                  target_ulong *args)

{

    target_ulong list = ppc64_phys_to_real(args[0]);

    target_ulong ov_table;

    PowerPCCPUClass *pcc = POWERPC_CPU_GET_CLASS(cpu_);

    CPUState *cs;

    bool cpu_match = false, cpu_update = true;

    unsigned old_cpu_version = cpu_->cpu_version;

    unsigned compat_lvl = 0, cpu_version = 0;

    unsigned max_lvl = get_compat_level(cpu_->max_compat);

    int counter;

    sPAPROptionVector *ov5_guest, *ov5_cas_old, *ov5_updates;



    /* Parse PVR list */

    for (counter = 0; counter < 512; ++counter) {

        uint32_t pvr, pvr_mask;



        pvr_mask = ldl_be_phys(&address_space_memory, list);

        list += 4;

        pvr = ldl_be_phys(&address_space_memory, list);

        list += 4;



        trace_spapr_cas_pvr_try(pvr);

        if (!max_lvl &&

            ((cpu_->env.spr[SPR_PVR] & pvr_mask) == (pvr & pvr_mask))) {

            cpu_match = true;

            cpu_version = 0;

        } else if (pvr == cpu_->cpu_version) {

            cpu_match = true;

            cpu_version = cpu_->cpu_version;

        } else if (!cpu_match) {

            cas_handle_compat_cpu(pcc, pvr, max_lvl, &compat_lvl, &cpu_version);

        }

        /* Terminator record */

        if (~pvr_mask & pvr) {

            break;

        }

    }



    /* Parsing finished */

    trace_spapr_cas_pvr(cpu_->cpu_version, cpu_match,

                        cpu_version, pcc->pcr_mask);



    /* Update CPUs */

    if (old_cpu_version != cpu_version) {

        CPU_FOREACH(cs) {

            SetCompatState s = {

                .cpu_version = cpu_version,

                .err = NULL,

            };



            run_on_cpu(cs, do_set_compat, RUN_ON_CPU_HOST_PTR(&s));



            if (s.err) {

                error_report_err(s.err);

                return H_HARDWARE;

            }

        }

    }



    if (!cpu_version) {

        cpu_update = false;

    }



    /* For the future use: here @ov_table points to the first option vector */

    ov_table = list;



    ov5_guest = spapr_ovec_parse_vector(ov_table, 5);



    /* NOTE: there are actually a number of ov5 bits where input from the

     * guest is always zero, and the platform/QEMU enables them independently

     * of guest input. To model these properly we'd want some sort of mask,

     * but since they only currently apply to memory migration as defined

     * by LoPAPR 1.1, 14.5.4.8, which QEMU doesn't implement, we don't need

     * to worry about this for now.

     */

    ov5_cas_old = spapr_ovec_clone(spapr->ov5_cas);

    /* full range of negotiated ov5 capabilities */

    spapr_ovec_intersect(spapr->ov5_cas, spapr->ov5, ov5_guest);

    spapr_ovec_cleanup(ov5_guest);

    /* capabilities that have been added since CAS-generated guest reset.

     * if capabilities have since been removed, generate another reset

     */

    ov5_updates = spapr_ovec_new();

    spapr->cas_reboot = spapr_ovec_diff(ov5_updates,

                                        ov5_cas_old, spapr->ov5_cas);



    if (!spapr->cas_reboot) {

        spapr->cas_reboot =

            (spapr_h_cas_compose_response(spapr, args[1], args[2], cpu_update,

                                          ov5_updates) != 0);

    }

    spapr_ovec_cleanup(ov5_updates);



    if (spapr->cas_reboot) {

        qemu_system_reset_request();

    }



    return H_SUCCESS;

}

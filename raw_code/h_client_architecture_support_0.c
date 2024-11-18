static target_ulong h_client_architecture_support(PowerPCCPU *cpu,

                                                  sPAPRMachineState *spapr,

                                                  target_ulong opcode,

                                                  target_ulong *args)

{

    /* Working address in data buffer */

    target_ulong addr = ppc64_phys_to_real(args[0]);

    target_ulong ov_table;

    uint32_t cas_pvr;

    sPAPROptionVector *ov1_guest, *ov5_guest, *ov5_cas_old, *ov5_updates;

    bool guest_radix;

    Error *local_err = NULL;

    bool raw_mode_supported = false;



    cas_pvr = cas_check_pvr(spapr, cpu, &addr, &raw_mode_supported, &local_err);

    if (local_err) {

        error_report_err(local_err);

        return H_HARDWARE;

    }



    /* Update CPUs */

    if (cpu->compat_pvr != cas_pvr) {

        ppc_set_compat_all(cas_pvr, &local_err);

        if (local_err) {

            /* We fail to set compat mode (likely because running with KVM PR),

             * but maybe we can fallback to raw mode if the guest supports it.

             */

            if (!raw_mode_supported) {

                error_report_err(local_err);

                return H_HARDWARE;

            }

            local_err = NULL;

        }

    }



    /* For the future use: here @ov_table points to the first option vector */

    ov_table = addr;



    ov1_guest = spapr_ovec_parse_vector(ov_table, 1);

    ov5_guest = spapr_ovec_parse_vector(ov_table, 5);

    if (spapr_ovec_test(ov5_guest, OV5_MMU_BOTH)) {

        error_report("guest requested hash and radix MMU, which is invalid.");

        exit(EXIT_FAILURE);

    }

    /* The radix/hash bit in byte 24 requires special handling: */

    guest_radix = spapr_ovec_test(ov5_guest, OV5_MMU_RADIX_300);

    spapr_ovec_clear(ov5_guest, OV5_MMU_RADIX_300);



    /*

     * HPT resizing is a bit of a special case, because when enabled

     * we assume an HPT guest will support it until it says it

     * doesn't, instead of assuming it won't support it until it says

     * it does.  Strictly speaking that approach could break for

     * guests which don't make a CAS call, but those are so old we

     * don't care about them.  Without that assumption we'd have to

     * make at least a temporary allocation of an HPT sized for max

     * memory, which could be impossibly difficult under KVM HV if

     * maxram is large.

     */

    if (!guest_radix && !spapr_ovec_test(ov5_guest, OV5_HPT_RESIZE)) {

        int maxshift = spapr_hpt_shift_for_ramsize(MACHINE(spapr)->maxram_size);



        if (spapr->resize_hpt == SPAPR_RESIZE_HPT_REQUIRED) {

            error_report(

                "h_client_architecture_support: Guest doesn't support HPT resizing, but resize-hpt=required");

            exit(1);

        }



        if (spapr->htab_shift < maxshift) {

            /* Guest doesn't know about HPT resizing, so we

             * pre-emptively resize for the maximum permitted RAM.  At

             * the point this is called, nothing should have been

             * entered into the existing HPT */

            spapr_reallocate_hpt(spapr, maxshift, &error_fatal);

            if (kvm_enabled()) {

                /* For KVM PR, update the HPT pointer */

                target_ulong sdr1 = (target_ulong)(uintptr_t)spapr->htab

                    | (spapr->htab_shift - 18);

                kvmppc_update_sdr1(sdr1);

            }

        }

    }



    /* NOTE: there are actually a number of ov5 bits where input from the

     * guest is always zero, and the platform/QEMU enables them independently

     * of guest input. To model these properly we'd want some sort of mask,

     * but since they only currently apply to memory migration as defined

     * by LoPAPR 1.1, 14.5.4.8, which QEMU doesn't implement, we don't need

     * to worry about this for now.

     */

    ov5_cas_old = spapr_ovec_clone(spapr->ov5_cas);



    /* also clear the radix/hash bit from the current ov5_cas bits to

     * be in sync with the newly ov5 bits. Else the radix bit will be

     * seen as being removed and this will generate a reset loop

     */

    spapr_ovec_clear(ov5_cas_old, OV5_MMU_RADIX_300);



    /* full range of negotiated ov5 capabilities */

    spapr_ovec_intersect(spapr->ov5_cas, spapr->ov5, ov5_guest);

    spapr_ovec_cleanup(ov5_guest);

    /* capabilities that have been added since CAS-generated guest reset.

     * if capabilities have since been removed, generate another reset

     */

    ov5_updates = spapr_ovec_new();

    spapr->cas_reboot = spapr_ovec_diff(ov5_updates,

                                        ov5_cas_old, spapr->ov5_cas);

    /* Now that processing is finished, set the radix/hash bit for the

     * guest if it requested a valid mode; otherwise terminate the boot. */

    if (guest_radix) {

        if (kvm_enabled() && !kvmppc_has_cap_mmu_radix()) {

            error_report("Guest requested unavailable MMU mode (radix).");

            exit(EXIT_FAILURE);

        }

        spapr_ovec_set(spapr->ov5_cas, OV5_MMU_RADIX_300);

    } else {

        if (kvm_enabled() && kvmppc_has_cap_mmu_radix()

            && !kvmppc_has_cap_mmu_hash_v3()) {

            error_report("Guest requested unavailable MMU mode (hash).");

            exit(EXIT_FAILURE);

        }

    }

    spapr->cas_legacy_guest_workaround = !spapr_ovec_test(ov1_guest,

                                                          OV1_PPC_3_00);

    if (!spapr->cas_reboot) {

        spapr->cas_reboot =

            (spapr_h_cas_compose_response(spapr, args[1], args[2],

                                          ov5_updates) != 0);

    }

    spapr_ovec_cleanup(ov5_updates);



    if (spapr->cas_reboot) {

        qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_RESET);

    } else {

        /* If ppc_spapr_reset() did not set up a HPT but one is necessary

         * (because the guest isn't going to use radix) then set it up here. */

        if ((spapr->patb_entry & PATBE1_GR) && !guest_radix) {

            /* legacy hash or new hash: */

            spapr_setup_hpt_and_vrma(spapr);

        }

    }



    return H_SUCCESS;

}

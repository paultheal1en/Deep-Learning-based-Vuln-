static target_ulong h_resize_hpt_prepare(PowerPCCPU *cpu,

                                         sPAPRMachineState *spapr,

                                         target_ulong opcode,

                                         target_ulong *args)

{

    target_ulong flags = args[0];

    int shift = args[1];

    sPAPRPendingHPT *pending = spapr->pending_hpt;

    uint64_t current_ram_size = MACHINE(spapr)->ram_size;

    int rc;



    if (spapr->resize_hpt == SPAPR_RESIZE_HPT_DISABLED) {

        return H_AUTHORITY;

    }



    if (!spapr->htab_shift) {

        /* Radix guest, no HPT */

        return H_NOT_AVAILABLE;

    }



    trace_spapr_h_resize_hpt_prepare(flags, shift);



    if (flags != 0) {

        return H_PARAMETER;

    }



    if (shift && ((shift < 18) || (shift > 46))) {

        return H_PARAMETER;

    }



    current_ram_size = pc_existing_dimms_capacity(&error_fatal);



    /* We only allow the guest to allocate an HPT one order above what

     * we'd normally give them (to stop a small guest claiming a huge

     * chunk of resources in the HPT */

    if (shift > (spapr_hpt_shift_for_ramsize(current_ram_size) + 1)) {

        return H_RESOURCE;

    }



    rc = kvmppc_resize_hpt_prepare(cpu, flags, shift);

    if (rc != -ENOSYS) {

        return resize_hpt_convert_rc(rc);

    }



    if (pending) {

        /* something already in progress */

        if (pending->shift == shift) {

            /* and it's suitable */

            if (pending->complete) {

                return pending->ret;

            } else {

                return H_LONG_BUSY_ORDER_100_MSEC;

            }

        }



        /* not suitable, cancel and replace */

        cancel_hpt_prepare(spapr);

    }



    if (!shift) {

        /* nothing to do */

        return H_SUCCESS;

    }



    /* start new prepare */



    pending = g_new0(sPAPRPendingHPT, 1);

    pending->shift = shift;

    pending->ret = H_HARDWARE;



    qemu_thread_create(&pending->thread, "sPAPR HPT prepare",

                       hpt_prepare_thread, pending, QEMU_THREAD_DETACHED);



    spapr->pending_hpt = pending;



    /* In theory we could estimate the time more accurately based on

     * the new size, but there's not much point */

    return H_LONG_BUSY_ORDER_100_MSEC;

}

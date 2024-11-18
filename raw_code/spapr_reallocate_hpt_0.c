static void spapr_reallocate_hpt(sPAPRMachineState *spapr, int shift,

                                 Error **errp)

{

    long rc;



    /* Clean up any HPT info from a previous boot */

    g_free(spapr->htab);

    spapr->htab = NULL;

    spapr->htab_shift = 0;

    close_htab_fd(spapr);



    rc = kvmppc_reset_htab(shift);

    if (rc < 0) {

        /* kernel-side HPT needed, but couldn't allocate one */

        error_setg_errno(errp, errno,

                         "Failed to allocate KVM HPT of order %d (try smaller maxmem?)",

                         shift);

        /* This is almost certainly fatal, but if the caller really

         * wants to carry on with shift == 0, it's welcome to try */

    } else if (rc > 0) {

        /* kernel-side HPT allocated */

        if (rc != shift) {

            error_setg(errp,

                       "Requested order %d HPT, but kernel allocated order %ld (try smaller maxmem?)",

                       shift, rc);

        }



        spapr->htab_shift = shift;

        kvmppc_kern_htab = true;

    } else {

        /* kernel-side HPT not needed, allocate in userspace instead */

        size_t size = 1ULL << shift;

        int i;



        spapr->htab = qemu_memalign(size, size);

        if (!spapr->htab) {

            error_setg_errno(errp, errno,

                             "Could not allocate HPT of order %d", shift);

            return;

        }



        memset(spapr->htab, 0, size);

        spapr->htab_shift = shift;

        kvmppc_kern_htab = false;



        for (i = 0; i < size / HASH_PTE_SIZE_64; i++) {

            DIRTY_HPTE(HPTE(spapr->htab, i));

        }

    }

}

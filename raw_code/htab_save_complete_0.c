static int htab_save_complete(QEMUFile *f, void *opaque)

{

    sPAPRMachineState *spapr = opaque;



    /* Iteration header */

    qemu_put_be32(f, 0);



    if (!spapr->htab) {

        int rc;



        assert(kvm_enabled());



        rc = spapr_check_htab_fd(spapr);

        if (rc < 0) {

            return rc;

        }



        rc = kvmppc_save_htab(f, spapr->htab_fd, MAX_KVM_BUF_SIZE, -1);

        if (rc < 0) {

            return rc;

        }

        close(spapr->htab_fd);

        spapr->htab_fd = -1;

    } else {

        htab_save_later_pass(f, spapr, -1);

    }



    /* End marker */

    qemu_put_be32(f, 0);

    qemu_put_be16(f, 0);

    qemu_put_be16(f, 0);



    return 0;

}

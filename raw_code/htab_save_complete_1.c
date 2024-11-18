static int htab_save_complete(QEMUFile *f, void *opaque)

{

    sPAPRMachineState *spapr = opaque;

    int fd;



    /* Iteration header */

    qemu_put_be32(f, 0);



    if (!spapr->htab) {

        int rc;



        assert(kvm_enabled());



        fd = get_htab_fd(spapr);

        if (fd < 0) {

            return fd;

        }



        rc = kvmppc_save_htab(f, fd, MAX_KVM_BUF_SIZE, -1);

        if (rc < 0) {

            return rc;

        }

        close_htab_fd(spapr);

    } else {

        if (spapr->htab_first_pass) {

            htab_save_first_pass(f, spapr, -1);

        }

        htab_save_later_pass(f, spapr, -1);

    }



    /* End marker */

    qemu_put_be32(f, 0);

    qemu_put_be16(f, 0);

    qemu_put_be16(f, 0);



    return 0;

}

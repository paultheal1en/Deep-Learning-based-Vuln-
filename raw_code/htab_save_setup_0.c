static int htab_save_setup(QEMUFile *f, void *opaque)

{

    sPAPRMachineState *spapr = opaque;



    /* "Iteration" header */

    qemu_put_be32(f, spapr->htab_shift);



    if (spapr->htab) {

        spapr->htab_save_index = 0;

        spapr->htab_first_pass = true;

    } else {

        assert(kvm_enabled());



        spapr->htab_fd = kvmppc_get_htab_fd(false);

        spapr->htab_fd_stale = false;

        if (spapr->htab_fd < 0) {

            fprintf(stderr, "Unable to open fd for reading hash table from KVM: %s\n",

                    strerror(errno));

            return -1;

        }

    }





    return 0;

}

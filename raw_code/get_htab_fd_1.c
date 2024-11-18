static int get_htab_fd(sPAPRMachineState *spapr)

{

    if (spapr->htab_fd >= 0) {

        return spapr->htab_fd;

    }



    spapr->htab_fd = kvmppc_get_htab_fd(false);

    if (spapr->htab_fd < 0) {

        error_report("Unable to open fd for reading hash table from KVM: %s",

                     strerror(errno));

    }



    return spapr->htab_fd;

}

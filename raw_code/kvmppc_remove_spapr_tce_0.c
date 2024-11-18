int kvmppc_remove_spapr_tce(void *table, int fd, uint32_t window_size)

{

    long len;



    if (fd < 0) {

        return -1;

    }



    len = (window_size / SPAPR_TCE_PAGE_SIZE)*sizeof(sPAPRTCE);

    if ((munmap(table, len) < 0) ||

        (close(fd) < 0)) {

        fprintf(stderr, "KVM: Unexpected error removing TCE table: %s",

                strerror(errno));

        /* Leak the table */

    }



    return 0;

}

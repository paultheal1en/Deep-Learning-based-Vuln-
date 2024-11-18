void kvmppc_read_hptes(ppc_hash_pte64_t *hptes, hwaddr ptex, int n)

{

    int fd, rc;

    int i;



    fd = kvmppc_get_htab_fd(false, ptex, &error_abort);



    i = 0;

    while (i < n) {

        struct kvm_get_htab_header *hdr;

        int m = n < HPTES_PER_GROUP ? n : HPTES_PER_GROUP;

        char buf[sizeof(*hdr) + m * HASH_PTE_SIZE_64];



        rc = read(fd, buf, sizeof(buf));

        if (rc < 0) {

            hw_error("kvmppc_read_hptes: Unable to read HPTEs");

        }



        hdr = (struct kvm_get_htab_header *)buf;

        while ((i < n) && ((char *)hdr < (buf + rc))) {

            int invalid = hdr->n_invalid;



            if (hdr->index != (ptex + i)) {

                hw_error("kvmppc_read_hptes: Unexpected HPTE index %"PRIu32

                         " != (%"HWADDR_PRIu" + %d", hdr->index, ptex, i);

            }



            memcpy(hptes + i, hdr + 1, HASH_PTE_SIZE_64 * hdr->n_valid);

            i += hdr->n_valid;



            if ((n - i) < invalid) {

                invalid = n - i;

            }

            memset(hptes + i, 0, invalid * HASH_PTE_SIZE_64);

            i += hdr->n_invalid;



            hdr = (struct kvm_get_htab_header *)

                ((char *)(hdr + 1) + HASH_PTE_SIZE_64 * hdr->n_valid);

        }

    }



    close(fd);

}

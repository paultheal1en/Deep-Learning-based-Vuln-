static int htab_load(QEMUFile *f, void *opaque, int version_id)

{

    sPAPRMachineState *spapr = opaque;

    uint32_t section_hdr;

    int fd = -1;



    if (version_id < 1 || version_id > 1) {

        error_report("htab_load() bad version");

        return -EINVAL;

    }



    section_hdr = qemu_get_be32(f);



    if (section_hdr) {

        Error *local_err;



        /* First section gives the htab size */

        spapr_reallocate_hpt(spapr, section_hdr, &local_err);

        if (local_err) {

            error_report_err(local_err);

            return -EINVAL;

        }

        return 0;

    }



    if (!spapr->htab) {

        assert(kvm_enabled());



        fd = kvmppc_get_htab_fd(true);

        if (fd < 0) {

            error_report("Unable to open fd to restore KVM hash table: %s",

                         strerror(errno));

        }

    }



    while (true) {

        uint32_t index;

        uint16_t n_valid, n_invalid;



        index = qemu_get_be32(f);

        n_valid = qemu_get_be16(f);

        n_invalid = qemu_get_be16(f);



        if ((index == 0) && (n_valid == 0) && (n_invalid == 0)) {

            /* End of Stream */

            break;

        }



        if ((index + n_valid + n_invalid) >

            (HTAB_SIZE(spapr) / HASH_PTE_SIZE_64)) {

            /* Bad index in stream */

            error_report(

                "htab_load() bad index %d (%hd+%hd entries) in htab stream (htab_shift=%d)",

                index, n_valid, n_invalid, spapr->htab_shift);

            return -EINVAL;

        }



        if (spapr->htab) {

            if (n_valid) {

                qemu_get_buffer(f, HPTE(spapr->htab, index),

                                HASH_PTE_SIZE_64 * n_valid);

            }

            if (n_invalid) {

                memset(HPTE(spapr->htab, index + n_valid), 0,

                       HASH_PTE_SIZE_64 * n_invalid);

            }

        } else {

            int rc;



            assert(fd >= 0);



            rc = kvmppc_load_htab_chunk(f, fd, index, n_valid, n_invalid);

            if (rc < 0) {

                return rc;

            }

        }

    }



    if (!spapr->htab) {

        assert(fd >= 0);

        close(fd);

    }



    return 0;

}

int spapr_vio_check_tces(VIOsPAPRDevice *dev, target_ulong ioba,

                         target_ulong len, enum VIOsPAPR_TCEAccess access)

{

    int start, end, i;



    start = ioba >> SPAPR_VIO_TCE_PAGE_SHIFT;

    end = (ioba + len - 1) >> SPAPR_VIO_TCE_PAGE_SHIFT;



    for (i = start; i <= end; i++) {

        if ((dev->rtce_table[i].tce & access) != access) {

#ifdef DEBUG_TCE

            fprintf(stderr, "FAIL on %d\n", i);

#endif

            return -1;

        }

    }



    return 0;

}

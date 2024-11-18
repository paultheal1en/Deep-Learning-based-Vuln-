static void trigger_page_fault(CPUS390XState *env, target_ulong vaddr,

                               uint32_t type, uint64_t asc, int rw, bool exc)

{

    int ilen = ILEN_LATER;

    uint64_t tec;



    tec = vaddr | (rw == MMU_DATA_STORE ? FS_WRITE : FS_READ) | asc >> 46;



    DPRINTF("%s: trans_exc_code=%016" PRIx64 "\n", __func__, tec);



    if (!exc) {

        return;

    }



    /* Code accesses have an undefined ilc.  */

    if (rw == MMU_INST_FETCH) {

        ilen = 2;

    }



    trigger_access_exception(env, type, ilen, tec);

}

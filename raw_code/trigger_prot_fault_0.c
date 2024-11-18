static void trigger_prot_fault(CPUS390XState *env, target_ulong vaddr,

                               uint64_t asc, int rw, bool exc)

{

    uint64_t tec;



    tec = vaddr | (rw == MMU_DATA_STORE ? FS_WRITE : FS_READ) | 4 | asc >> 46;



    DPRINTF("%s: trans_exc_code=%016" PRIx64 "\n", __func__, tec);



    if (!exc) {

        return;

    }



    trigger_access_exception(env, PGM_PROTECTION, ILEN_LATER_INC, tec);

}

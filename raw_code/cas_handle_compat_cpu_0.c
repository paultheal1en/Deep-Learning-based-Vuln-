static void cas_handle_compat_cpu(PowerPCCPUClass *pcc, uint32_t pvr,

                                  unsigned max_lvl, unsigned *compat_lvl,

                                  unsigned *cpu_version)

{

    unsigned lvl = get_compat_level(pvr);

    bool is205, is206;



    if (!lvl) {

        return;

    }



    /* If it is a logical PVR, try to determine the highest level */

    is205 = (pcc->pcr_mask & PCR_COMPAT_2_05) &&

            (lvl == get_compat_level(CPU_POWERPC_LOGICAL_2_05));

    is206 = (pcc->pcr_mask & PCR_COMPAT_2_06) &&

            ((lvl == get_compat_level(CPU_POWERPC_LOGICAL_2_06)) ||

             (lvl == get_compat_level(CPU_POWERPC_LOGICAL_2_06_PLUS)));



    if (is205 || is206) {

        if (!max_lvl) {

            /* User did not set the level, choose the highest */

            if (*compat_lvl <= lvl) {

                *compat_lvl = lvl;

                *cpu_version = pvr;

            }

        } else if (max_lvl >= lvl) {

            /* User chose the level, don't set higher than this */

            *compat_lvl = lvl;

            *cpu_version = pvr;

        }

    }

}

static CPAccessResult ats_access(CPUARMState *env, const ARMCPRegInfo *ri)

{

    if (ri->opc2 & 4) {

        /* Other states are only available with TrustZone; in

         * a non-TZ implementation these registers don't exist

         * at all, which is an Uncategorized trap. This underdecoding

         * is safe because the reginfo is NO_MIGRATE.

         */

        return CP_ACCESS_TRAP_UNCATEGORIZED;

    }

    return CP_ACCESS_OK;

}

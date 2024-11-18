static void add_cpreg_to_hashtable(ARMCPU *cpu, const ARMCPRegInfo *r,

                                   void *opaque, int state, int secstate,

                                   int crm, int opc1, int opc2)

{

    /* Private utility function for define_one_arm_cp_reg_with_opaque():

     * add a single reginfo struct to the hash table.

     */

    uint32_t *key = g_new(uint32_t, 1);

    ARMCPRegInfo *r2 = g_memdup(r, sizeof(ARMCPRegInfo));

    int is64 = (r->type & ARM_CP_64BIT) ? 1 : 0;

    int ns = (secstate & ARM_CP_SECSTATE_NS) ? 1 : 0;



    /* Reset the secure state to the specific incoming state.  This is

     * necessary as the register may have been defined with both states.

     */

    r2->secure = secstate;



    if (r->bank_fieldoffsets[0] && r->bank_fieldoffsets[1]) {

        /* Register is banked (using both entries in array).

         * Overwriting fieldoffset as the array is only used to define

         * banked registers but later only fieldoffset is used.

         */

        r2->fieldoffset = r->bank_fieldoffsets[ns];

    }



    if (state == ARM_CP_STATE_AA32) {

        if (r->bank_fieldoffsets[0] && r->bank_fieldoffsets[1]) {

            /* If the register is banked then we don't need to migrate or

             * reset the 32-bit instance in certain cases:

             *

             * 1) If the register has both 32-bit and 64-bit instances then we

             *    can count on the 64-bit instance taking care of the

             *    non-secure bank.

             * 2) If ARMv8 is enabled then we can count on a 64-bit version

             *    taking care of the secure bank.  This requires that separate

             *    32 and 64-bit definitions are provided.

             */

            if ((r->state == ARM_CP_STATE_BOTH && ns) ||

                (arm_feature(&cpu->env, ARM_FEATURE_V8) && !ns)) {

                r2->type |= ARM_CP_NO_MIGRATE;

                r2->resetfn = arm_cp_reset_ignore;

            }

        } else if ((secstate != r->secure) && !ns) {

            /* The register is not banked so we only want to allow migration of

             * the non-secure instance.

             */

            r2->type |= ARM_CP_NO_MIGRATE;

            r2->resetfn = arm_cp_reset_ignore;

        }



        if (r->state == ARM_CP_STATE_BOTH) {

            /* We assume it is a cp15 register if the .cp field is left unset.

             */

            if (r2->cp == 0) {

                r2->cp = 15;

            }



#ifdef HOST_WORDS_BIGENDIAN

            if (r2->fieldoffset) {

                r2->fieldoffset += sizeof(uint32_t);

            }

#endif

        }

    }

    if (state == ARM_CP_STATE_AA64) {

        /* To allow abbreviation of ARMCPRegInfo

         * definitions, we treat cp == 0 as equivalent to

         * the value for "standard guest-visible sysreg".

         * STATE_BOTH definitions are also always "standard

         * sysreg" in their AArch64 view (the .cp value may

         * be non-zero for the benefit of the AArch32 view).

         */

        if (r->cp == 0 || r->state == ARM_CP_STATE_BOTH) {

            r2->cp = CP_REG_ARM64_SYSREG_CP;

        }

        *key = ENCODE_AA64_CP_REG(r2->cp, r2->crn, crm,

                                  r2->opc0, opc1, opc2);

    } else {

        *key = ENCODE_CP_REG(r2->cp, is64, ns, r2->crn, crm, opc1, opc2);

    }

    if (opaque) {

        r2->opaque = opaque;

    }

    /* reginfo passed to helpers is correct for the actual access,

     * and is never ARM_CP_STATE_BOTH:

     */

    r2->state = state;

    /* Make sure reginfo passed to helpers for wildcarded regs

     * has the correct crm/opc1/opc2 for this reg, not CP_ANY:

     */

    r2->crm = crm;

    r2->opc1 = opc1;

    r2->opc2 = opc2;

    /* By convention, for wildcarded registers only the first

     * entry is used for migration; the others are marked as

     * NO_MIGRATE so we don't try to transfer the register

     * multiple times. Special registers (ie NOP/WFI) are

     * never migratable.

     */

    if ((r->type & ARM_CP_SPECIAL) ||

        ((r->crm == CP_ANY) && crm != 0) ||

        ((r->opc1 == CP_ANY) && opc1 != 0) ||

        ((r->opc2 == CP_ANY) && opc2 != 0)) {

        r2->type |= ARM_CP_NO_MIGRATE;

    }



    /* Overriding of an existing definition must be explicitly

     * requested.

     */

    if (!(r->type & ARM_CP_OVERRIDE)) {

        ARMCPRegInfo *oldreg;

        oldreg = g_hash_table_lookup(cpu->cp_regs, key);

        if (oldreg && !(oldreg->type & ARM_CP_OVERRIDE)) {

            fprintf(stderr, "Register redefined: cp=%d %d bit "

                    "crn=%d crm=%d opc1=%d opc2=%d, "

                    "was %s, now %s\n", r2->cp, 32 + 32 * is64,

                    r2->crn, r2->crm, r2->opc1, r2->opc2,

                    oldreg->name, r2->name);

            g_assert_not_reached();

        }

    }

    g_hash_table_insert(cpu->cp_regs, key, r2);

}
